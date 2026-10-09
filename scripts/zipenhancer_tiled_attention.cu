// Controlled ZipEnhancer inference experiment; not a production replacement.
// Build as a CUDA DLL, e.g. nvcc -O3 -std=c++17 -arch=sm_120 --shared
//   -Xcompiler /MT zipenhancer_tiled_attention.cu -o zipenhancer_tiled_attention.dll
// Requires WMMA-capable CUDA hardware (SM70+); use the actual target architecture.
// Do not use --use_fast_math for the initial numerical/performance baseline.
//
// Math follows zipformer.py RelPositionMultiheadAttentionWeights: no sqrt(D)
// scale, and positional index N-1-query+key (the source as_strided mapping).
// Numeric policy: FP16 inputs, FP32 QK WMMA and four-term positional accumulation;
// round each dot to half, then round their sum to half, matching source AMP's
// logit boundaries. Dot reduction order may still differ from cuBLAS. Stable
// online softmax statistics and value accumulation stay FP32. Unnormalized
// tile probabilities round to half for Tensor Core P*V; this differs from the
// source's half-rounded normalized probabilities and requires waveform admission.
// NonlinAttention uses 3*embed_dim/4 channels: 48 for the original embed_dim=64.
//
// Each CTA owns 16 query rows, streaming keys in tiles of 64. Both QK and
// probability*V use WMMA with FP32 accumulation. Shared score rows are skewed
// so adjacent 8-lane query groups do not conflict on the same memory banks.
// Scratch is bounded shared memory: no N*N tensors, heap allocation, host copies,
// default-stream work, or host synchronization.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <mma.h>

#include <climits>
#include <cstddef>

#if defined(_WIN32)
#define ZIP_EXPORT extern "C" __declspec(dllexport)
#else
#define ZIP_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

constexpr int kQueryTile = 16;
constexpr int kThreads = 128;
constexpr int kKeyTile = 64;
constexpr int kScoreStride = 72;
constexpr int kDotStride = 16;
constexpr unsigned kWarpMask = 0xffffffffu;

__device__ __forceinline__ float row_max(float value) {
#pragma unroll
    for (int delta = 4; delta > 0; delta >>= 1) {
        value = fmaxf(value, __shfl_down_sync(kWarpMask, value, delta, 8));
    }
    return __shfl_sync(kWarpMask, value, 0, 8);
}

__device__ __forceinline__ float row_sum(float value) {
#pragma unroll
    for (int delta = 4; delta > 0; delta >>= 1) {
        value += __shfl_down_sync(kWarpMask, value, delta, 8);
    }
    return __shfl_sync(kWarpMask, value, 0, 8);
}

template <int ValueStride>
__global__ void tiled_attention(
    const half* q, const half* k, const half* p, const half* pos,
    const half* v, half* out, int batch, int length, int value_dim) {
    // All WMMA load/store bases are 32-byte aligned; half ldm=16 and float
    // ldm=72 obey WMMA alignment/stride requirements, including warp offsets.
    __shared__ __align__(32) half query[kQueryTile * kDotStride];
    __shared__ __align__(32) half keys[kKeyTile * kDotStride];
    __shared__ __align__(32) half position_query[kQueryTile * 4];
    __shared__ __align__(32) half values[kKeyTile * ValueStride];
    __shared__ __align__(32) float scores[kQueryTile * kScoreStride];
    __shared__ __align__(32) half probabilities[kQueryTile * kKeyTile];

    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int row = tid / 8;
    const int column_lane = tid % 8;
    const int query_start = static_cast<int>(blockIdx.x) * kQueryTile;
    const int query_index = query_start + row;
    // q/k/p/v/out are [head,batch,N,dim]; pos is [head,2*N-1,4].
    const std::size_t head_batch = blockIdx.y;
    const std::size_t head = head_batch / static_cast<std::size_t>(batch);
    const std::size_t sequence_base = head_batch * static_cast<std::size_t>(length);
    const std::size_t pos_base = head * (2 * static_cast<std::size_t>(length) - 1) * 4;
    const half zero = __float2half_rn(0.0f);

    for (int index = tid; index < kQueryTile * kDotStride; index += kThreads) {
        const int source_row = query_start + index / kDotStride;
        const int channel = index % kDotStride;
        query[index] = source_row < length && channel < 12
            ? q[(sequence_base + source_row) * 12 + channel] : zero;
    }
    if (tid < kQueryTile * 4) {
        const int source_row = query_start + tid / 4;
        position_query[tid] = source_row < length
            ? p[(sequence_base + source_row) * 4 + tid % 4] : zero;
    }
    __syncthreads();

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                           half, nvcuda::wmma::row_major> query_fragment;
    nvcuda::wmma::load_matrix_sync(query_fragment, query, kDotStride);

    float accumulated[ValueStride / 8] = {};
    float running_max = -CUDART_INF_F;
    float denominator = 0.0f;
    const float p0 = __half2float(position_query[row * 4]);
    const float p1 = __half2float(position_query[row * 4 + 1]);
    const float p2 = __half2float(position_query[row * 4 + 2]);
    const float p3 = __half2float(position_query[row * 4 + 3]);

    for (int key_start = 0; key_start < length; key_start += kKeyTile) {
        const int key_count = length - key_start < kKeyTile
            ? length - key_start : kKeyTile;
        for (int index = tid; index < kKeyTile * kDotStride; index += kThreads) {
            const int source_row = key_start + index / kDotStride;
            const int channel = index % kDotStride;
            keys[index] = source_row < length && channel < 12
                ? k[(sequence_base + source_row) * 12 + channel] : zero;
        }
        for (int index = tid; index < kKeyTile * ValueStride; index += kThreads) {
            const int source_row = key_start + index / ValueStride;
            const int channel = index % ValueStride;
            values[index] = source_row < length && channel < value_dim
                ? v[(sequence_base + source_row) * value_dim + channel] : zero;
        }
        __syncthreads();

        // Four warps independently produce the four 16-column score subtiles.
        // keys[key][channel] is the column-major representation of K transpose.
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                               half, nvcuda::wmma::col_major> key_fragment;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16,
                               float> score_fragment;
        nvcuda::wmma::load_matrix_sync(
            key_fragment, keys + warp * 16 * kDotStride, kDotStride);
        nvcuda::wmma::fill_fragment(score_fragment, 0.0f);
        nvcuda::wmma::mma_sync(
            score_fragment, query_fragment, key_fragment, score_fragment);
        nvcuda::wmma::store_matrix_sync(
            scores + warp * 16, score_fragment, kScoreStride, nvcuda::wmma::mem_row_major);
        __syncthreads();

        // Eight lanes cooperate on each query row. Padded query rows still take
        // part in every shuffle/barrier, but neither read source rows nor write
        // output. Only keys beyond N are masked; zero-valued keys inside N stay.
        float logits[kKeyTile / 8];
        float tile_max = -CUDART_INF_F;
#pragma unroll
        for (int slot = 0; slot < kKeyTile / 8; ++slot) {
            const int column = column_lane + slot * 8;
            float logit = -CUDART_INF_F;
            if (column < key_count) {
                logit = 0.0f;
                if (query_index < length) {
                    const int relative_index = length - 1 - query_index + key_start + column;
                    const half* relative = pos + pos_base + static_cast<std::size_t>(relative_index) * 4;
                    float positional = p0 * __half2float(relative[0]);
                    positional = fmaf(p1, __half2float(relative[1]), positional);
                    positional = fmaf(p2, __half2float(relative[2]), positional);
                    positional = fmaf(p3, __half2float(relative[3]), positional);
                    const float content_half = __half2float(
                        __float2half_rn(scores[row * kScoreStride + column]));
                    const float position_half = __half2float(__float2half_rn(positional));
                    logit = __half2float(__float2half_rn(content_half + position_half));
                }
            }
            logits[slot] = logit;
            tile_max = fmaxf(tile_max, logit);
        }
        tile_max = row_max(tile_max);
        const float new_max = fmaxf(running_max, tile_max);
        const float rescale = expf(running_max - new_max);
        float tile_sum = 0.0f;
#pragma unroll
        for (int slot = 0; slot < kKeyTile / 8; ++slot) {
            const int column = column_lane + slot * 8;
            const float weight = expf(logits[slot] - new_max);
            probabilities[row * kKeyTile + column] = __float2half_rn(weight);
            tile_sum += weight;
        }
        denominator = denominator * rescale + row_sum(tile_sum);
        running_max = new_max;
        __syncthreads();

        if (warp < ValueStride / 16) {
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16,
                                   float> value_result;
            nvcuda::wmma::fill_fragment(value_result, 0.0f);
#pragma unroll
            for (int key_offset = 0; key_offset < kKeyTile; key_offset += 16) {
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                                       half, nvcuda::wmma::row_major> probability_fragment;
                nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                                       half, nvcuda::wmma::row_major> value_fragment;
                nvcuda::wmma::load_matrix_sync(probability_fragment,
                    probabilities + key_offset, kKeyTile);
                nvcuda::wmma::load_matrix_sync(value_fragment,
                    values + key_offset * ValueStride + warp * 16, ValueStride);
                nvcuda::wmma::mma_sync(value_result, probability_fragment,
                    value_fragment, value_result);
            }
            nvcuda::wmma::store_matrix_sync(scores + warp * 16, value_result,
                kScoreStride, nvcuda::wmma::mem_row_major);
        }
        __syncthreads();
#pragma unroll
        for (int slot = 0; slot < ValueStride / 8; ++slot) {
            const int channel = column_lane + slot * 8;
            if (channel < value_dim) {
                accumulated[slot] = accumulated[slot] * rescale +
                    scores[row * kScoreStride + channel];
            }
        }
        // Protect all shared tile readers before the next key tile overwrites.
        __syncthreads();
    }

    if (query_index < length) {
        const float inverse_denominator = 1.0f / denominator;
#pragma unroll
        for (int slot = 0; slot < ValueStride / 8; ++slot) {
            const int channel = column_lane + slot * 8;
            if (channel < value_dim) {
                out[(sequence_base + query_index) * value_dim + channel] =
                    __float2half_rn(accumulated[slot] * inverse_denominator);
            }
        }
    }
}

template <int ValueStride>
inline void launch_attention(const half* q, const half* k, const half* p,
    const half* pos, const half* v, half* out, int heads, int batch,
    int length, int value_dim, cudaStream_t stream) {
    const dim3 grid((length + kQueryTile - 1) / kQueryTile, heads * batch);
    tiled_attention<ValueStride><<<grid, kThreads, 0, stream>>>(
        q, k, p, pos, v, out, batch, length, value_dim);
}

}  // namespace

// Buffers must be contiguous device FP16 tensors on the caller's current CUDA
// device, with non-overlapping output. stream is the caller's cudaStream_t.
// The launch is asynchronous/capture-safe: return launch status, not a device
// execution fence. The caller must surface asynchronous errors at its own fence.
ZIP_EXPORT int zip_attention(
    const void* q, const void* k, const void* p, const void* pos,
    const void* v, void* out, int heads, int batch, int length,
    int queryDim, int posDim, int valueDim, void* stream) {
    // Limit grid.y to CUDA's portable limit and keep signed relative/key-tile
    // arithmetic in range. Tensor byte offsets themselves use size_t.
    if (!q || !k || !p || !pos || !v || !out ||
        (heads != 1 && heads != 4) || batch <= 0 || batch > 65535 / heads ||
        length <= 0 || length > INT_MAX / 2 || queryDim != 12 || posDim != 4 ||
        valueDim < 8 || valueDim > 64) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const half* query = static_cast<const half*>(q);
    const half* key = static_cast<const half*>(k);
    const half* position_query = static_cast<const half*>(p);
    const half* positions = static_cast<const half*>(pos);
    const half* value = static_cast<const half*>(v);
    half* output = static_cast<half*>(out);
    if (valueDim <= 16) {
        launch_attention<16>(query, key, position_query, positions,
            value, output, heads, batch, length, valueDim, cuda_stream);
    } else if (valueDim <= 32) {
        launch_attention<32>(query, key, position_query, positions,
            value, output, heads, batch, length, valueDim, cuda_stream);
    } else if (valueDim <= 48) {
        launch_attention<48>(query, key, position_query, positions,
            value, output, heads, batch, length, valueDim, cuda_stream);
    } else {
        launch_attention<64>(query, key, position_query, positions,
            value, output, heads, batch, length, valueDim, cuda_stream);
    }
    return static_cast<int>(cudaGetLastError());
}

ZIP_EXPORT const char* zip_attention_error(int status) {
    return cudaGetErrorString(static_cast<cudaError_t>(status));
}
