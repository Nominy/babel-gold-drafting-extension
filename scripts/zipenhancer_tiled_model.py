"""Controlled, inference-only CUDA attention experiment for original ZipEnhancer.

The descriptor replaces only materialized attention weights: every query still
attends to every key with the original relative-position indexing. Projection,
nonlinear, diagnostic-identity and output modules retain their pretrained objects.
Run under inference_mode/no_grad with CUDA FP16 autocast (or FP16 model/inputs).
Install before torch.compile or CUDA graph warmup/capture; no DLL is loaded during
forward. The caller retains the usual responsibility for cross-stream readiness.
The kernel rounds both score dot products and their sum to FP16. Online softmax
statistics and weighted-value accumulation stay FP32; unnormalized tile
probabilities round to FP16 for Tensor Core multiplication. This differs from
the original half in-place softmax and requires full-wave numerical admission.
"""
import ctypes
import hashlib
from pathlib import Path

import torch
from torch import nn


_EXPECTED = {"heads": 4, "query_dim": 12, "pos_head_dim": 4,
             "embed_dim": 64, "pos_input_dim": 24, "value_dim": 8,
             "nonlinear_value_dim": 48}
_LIBRARIES = {}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_attention_inputs(q, k, p, pos, v):
    """Metadata-only validation, also used by the fake implementation."""
    for name, tensor, rank in (("q", q, 4), ("k", k, 4), ("p", p, 4),
                               ("pos", pos, 3), ("v", v, 4)):
        if tensor.layout != torch.strided or tensor.ndim != rank:
            raise ValueError(f"Tiled attention {name} must be a rank-{rank} strided tensor")
        if tensor.device.type != "cuda" or tensor.dtype != torch.float16:
            raise TypeError(f"Tiled attention {name} must be CUDA FP16; no dtype/device conversion is performed")
        if tensor.device != q.device:
            raise ValueError("Tiled attention inputs must share one CUDA device")
        if tensor.requires_grad:
            raise ValueError("Tiled attention is inference-only; use torch.inference_mode() or torch.no_grad()")
    heads, batch, length, query_dim = q.shape
    if heads not in (1, 4) or query_dim != 12 or batch < 1 or length < 1:
        raise ValueError("Tiled attention requires H=1 or 4, Dq=12, and positive batch/sequence lengths")
    if batch > 2147483647 or length > 1073741824:
        raise ValueError("Tiled attention dimensions exceed the C ABI int range")
    if k.shape != q.shape or p.shape != (heads, batch, length, 4):
        raise ValueError("Tiled attention requires k[H,B,N,12] and p[H,B,N,4]")
    if pos.shape != (heads, 2 * length - 1, 4):
        raise ValueError("Tiled attention requires shared positions pos[H,2*N-1,4]")
    if v.shape[:3] != (heads, batch, length) or not 8 <= v.shape[3] <= 64:
        raise ValueError("Tiled attention requires v[H,B,N,Dv] with 8 <= Dv <= 64")


class _AttentionLibrary:
    def __init__(self, path, digest):
        self.path, self.sha256 = path, digest
        if ctypes.sizeof(ctypes.c_void_p) != 8:
            raise RuntimeError("Tiled attention requires a 64-bit Python process")
        try:
            # The ABI is extern C/cdecl, not WinDLL's stdcall. Never search by basename.
            self.handle = ctypes.CDLL(str(path))
            self.launch = self.handle.zip_attention
            self.error = self.handle.zip_attention_error
        except (OSError, AttributeError) as exc:
            raise RuntimeError(f"Cannot load tiled attention DLL/exports at {path}: {exc}") from exc
        self.launch.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 6 + [ctypes.c_void_p]
        self.launch.restype = ctypes.c_int
        self.error.argtypes = [ctypes.c_int]
        self.error.restype = ctypes.c_char_p
        if _sha256(path) != digest:
            raise RuntimeError(f"Tiled attention DLL changed while loading: {path}")
        self.op_name = f"zipenhancer_tiled::attention_{digest}"
        self.op = self._register_op()

    def _register_op(self):
        library = self

        @torch.library.custom_op(self.op_name, mutates_args=(), device_types="cuda")
        def attention(q: torch.Tensor, k: torch.Tensor, p: torch.Tensor,
                      pos: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            _check_attention_inputs(q, k, p, pos, v)
            with torch.cuda.device(q.device):
                stream = torch.cuda.current_stream(q.device)
                buffers = []
                for tensor in (q, k, p, pos, v):
                    tensor.record_stream(stream)
                    dense = tensor.contiguous()
                    if dense is not tensor:
                        dense.record_stream(stream)
                    buffers.append(dense)
                q, k, p, pos, v = buffers
                # Each invocation owns its output, including during graph capture.
                # There is no shared scratch/output address to alias another call.
                output = torch.empty_like(v, memory_format=torch.contiguous_format)
                output.record_stream(stream)
                heads, batch, length, query_dim = q.shape
                status = library.launch(
                    *(ctypes.c_void_p(tensor.data_ptr()) for tensor in (*buffers, output)),
                    heads, batch, length, query_dim, p.shape[3], v.shape[3],
                    ctypes.c_void_p(stream.cuda_stream),
                )
                if status:
                    detail = library.error(status)
                    message = detail.decode("utf-8", errors="replace") if detail else "unknown CUDA error"
                    raise RuntimeError(f"zip_attention failed ({status}: {message}); "
                                       f"H={heads}, B={batch}, N={length}, Dv={v.shape[3]}, DLL={library.path}")
                return output

        @attention.register_fake
        def fake(q, k, p, pos, v):
            _check_attention_inputs(q, k, p, pos, v)
            return torch.empty_like(v, memory_format=torch.contiguous_format)

        return attention


def _load_library(library_path):
    path = Path(library_path).expanduser().resolve(strict=True)
    if not path.is_file() or path.suffix.lower() != ".dll":
        raise ValueError(f"An explicit CUDA DLL file is required: {path}")
    digest = _sha256(path)
    cached = _LIBRARIES.get(path)
    if cached is not None:
        if cached.sha256 != digest:
            raise RuntimeError("A loaded attention DLL changed on disk; restart Python before loading another build")
        return cached
    # One registration per binary, even when the same DLL has been copied.
    for library in _LIBRARIES.values():
        if library.sha256 == digest:
            _LIBRARIES[path] = library
            return library
    library = _AttentionLibrary(path, digest)
    _LIBRARIES[path] = library  # Keep native code and its custom-op registration alive.
    return library


class AttentionPlan:
    """Linear-size projected inputs; shape is metadata, never an N-by-N tensor."""

    def __init__(self, q, k, p, pos, op):
        self.q, self.k, self.p, self.pos = q, k, p, pos
        self.op = op
        heads, batch, length, _ = q.shape
        self.shape = (heads, batch, length, length)

    def __getitem__(self, selection):
        # Zipformer2EncoderLayer.forward selects [0:1] for NonlinAttention.
        if not isinstance(selection, slice) or selection.start != 0 or selection.stop != 1 or selection.step not in (None, 1):
            raise ValueError("Tiled attention supports only the original first-head slice [0:1]")
        return AttentionPlan(self.q[0:1], self.k[0:1], self.p[0:1], self.pos[0:1], self.op)

    def apply(self, values):
        return self.op(self.q, self.k, self.p, self.pos, values)


def _check_inference(wrapper):
    if wrapper.training or wrapper.original.training:
        raise ValueError("Tiled attention requires eval mode; training is unsupported")
    if torch.is_grad_enabled():
        raise ValueError("Tiled attention requires torch.inference_mode() or torch.no_grad()")


def _check_activation(value, name):
    if value.ndim != 3 or value.device.type != "cuda":
        raise ValueError(f"Tiled attention {name} must be a rank-3 CUDA tensor")
    if value.dtype == torch.float16:
        return
    if (value.dtype == torch.float32 and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.float16):
        return  # Preserve the original Linear's AMP conversion, never cast here.
    raise TypeError(f"Tiled attention {name} requires FP16, or FP32 inside CUDA FP16 autocast")


class TiledAttentionWeights(nn.Module):
    def __init__(self, original, op):
        super().__init__()
        self.original, self.op = original, op
        self.training = False

    def forward(self, x, pos_emb, key_padding_mask=None, attn_mask=None):
        _check_inference(self)
        if key_padding_mask is not None or attn_mask is not None:
            raise ValueError("Tiled attention does not support masks; they must not be silently ignored")
        _check_activation(x, "input")
        _check_activation(pos_emb, "positional embedding")
        original = self.original
        length, batch, embed_dim = x.shape
        if embed_dim != original.embed_dim:
            raise ValueError("Unexpected tiled attention input embedding dimension")
        if pos_emb.shape != (1, 2 * length - 1, original.linear_pos.in_features):
            raise ValueError("Tiled attention requires one shared positional-embedding batch")
        x = original.in_proj(x)
        heads, query_dim, pos_dim = original.num_heads, original.query_head_dim, original.pos_head_dim
        width = heads * query_dim
        q = original.copy_query(x[..., :width])
        k = original.whiten_keys(original.balance_keys(x[..., width:2 * width]))
        p = original.copy_pos_query(x[..., 2 * width:])
        q = q.reshape(length, batch, heads, query_dim).permute(2, 1, 0, 3).contiguous()
        k = k.reshape(length, batch, heads, query_dim).permute(2, 1, 0, 3).contiguous()
        p = p.reshape(length, batch, heads, pos_dim).permute(2, 1, 0, 3).contiguous()
        positions = original.linear_pos(pos_emb)
        positions = positions.reshape(2 * length - 1, heads, pos_dim).permute(1, 0, 2).contiguous()
        if any(tensor.dtype != torch.float16 for tensor in (q, k, p, positions)):
            raise TypeError("Original attention projections must produce FP16; use CUDA FP16 autocast")
        # Original relative column = N - 1 - query_index + key_index, no extra
        # 1/sqrt(D) scaling. Eval dropout and positional skipping are both no-ops.
        return AttentionPlan(q, k, p, positions, self.op)


class TiledSelfAttention(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
        self.training = False

    def forward(self, x, attn_weights):
        _check_inference(self)
        _check_activation(x, "self-attention input")
        if not isinstance(attn_weights, AttentionPlan):
            raise TypeError("Tiled SelfAttention requires an AttentionPlan")
        length, batch, _ = x.shape
        heads = attn_weights.shape[0]
        if heads != 4 or attn_weights.shape != (heads, batch, length, length):
            raise ValueError("Tiled SelfAttention input and attention-plan shapes differ")
        original = self.original
        x = original.in_proj(x)
        x = x.reshape(length, batch, heads, -1).permute(2, 1, 0, 3)
        value_dim = x.shape[-1]
        x = attn_weights.apply(x)
        x = x.permute(2, 1, 0, 3).contiguous().view(length, batch, heads * value_dim)
        return original.whiten(original.out_proj(x))


class TiledNonlinAttention(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original
        self.training = False

    def forward(self, x, attn_weights):
        _check_inference(self)
        _check_activation(x, "nonlinear attention input")
        if not isinstance(attn_weights, AttentionPlan):
            raise TypeError("Tiled NonlinAttention requires an AttentionPlan")
        original = self.original
        x = original.in_proj(x)
        length, batch, _ = x.shape
        heads = attn_weights.shape[0]
        if heads != 1 or attn_weights.shape != (heads, batch, length, length):
            raise ValueError("Tiled NonlinAttention requires the original first-head plan")
        s, x, y = x.chunk(3, dim=2)
        s = original.balancer(s)
        s = original.tanh(s)
        s = s.unsqueeze(-1).reshape(length, batch, original.hidden_channels)
        x = original.whiten1(x)
        x = x * s
        x = original.identity1(x)
        x = x.reshape(length, batch, heads, -1).permute(2, 1, 0, 3)
        x = attn_weights.apply(x)
        x = x.permute(2, 1, 0, 3).reshape(length, batch, -1)
        y = original.identity2(y)
        x = x * y
        x = original.identity3(x)
        x = original.out_proj(x)
        return original.whiten2(x)


def _check_linear(module, name, inputs, outputs, bias):
    linear = getattr(module, name)
    if (not isinstance(linear, nn.Linear) or linear.in_features != inputs
            or linear.out_features != outputs or (linear.bias is not None) != bias):
        raise ValueError(f"Unsupported {type(module).__name__}.{name}: expected Linear({inputs}, {outputs}, bias={bias})")


def install_tiled_attention(model, library_path: Path) -> dict:
    """Replace the three original attention classes, preserving parameter objects.

    Metadata records every replaced path/class and SHA-256 of the loaded binary,
    CUDA source and this helper. No pretrained weight, key context or window is
    changed. The binary must already be built; no compiler/fallback is invoked.
    """
    from modelscope.models.audio.ans.zipenhancer_layers.zipformer import (
        RelPositionMultiheadAttentionWeights, SelfAttention, NonlinAttention,
    )

    if any(module.training for module in model.modules()):
        raise ValueError("Call model.eval() before installing tiled attention")
    if any(isinstance(module, (TiledAttentionWeights, TiledSelfAttention, TiledNonlinAttention))
           for module in model.modules()):
        raise ValueError("Tiled attention is already installed on this model")
    types = (RelPositionMultiheadAttentionWeights, SelfAttention, NonlinAttention)
    pending = []
    counts = {kind.__name__: 0 for kind in types}
    for name, module in model.named_modules():
        if not isinstance(module, types):
            continue
        if type(module) not in types:
            raise ValueError(f"Unsupported attention subclass at {name}: {type(module).__name__}")
        if not name:
            raise ValueError("Install tiled attention on the parent model, not a standalone attention module")
        if "forward" in module.__dict__:
            raise ValueError(f"Cannot replace modified original attention forward: {name}")
        if isinstance(module, RelPositionMultiheadAttentionWeights):
            if (module.embed_dim, module.num_heads, module.query_head_dim, module.pos_head_dim) != (64, 4, 12, 4):
                raise ValueError(f"Unsupported attention configuration at {name}; expected embed=64,H=4,Dq=12,Dp=4")
            _check_linear(module, "in_proj", 64, 112, True)
            _check_linear(module, "linear_pos", 24, 16, False)
            replacement = TiledAttentionWeights
        elif isinstance(module, SelfAttention):
            _check_linear(module, "in_proj", 64, 32, True)
            _check_linear(module, "out_proj", 32, 64, True)
            replacement = TiledSelfAttention
        else:
            if module.hidden_channels != 48:
                raise ValueError(f"Unsupported nonlinear attention width at {name}; expected 48")
            _check_linear(module, "in_proj", 64, 144, True)
            _check_linear(module, "out_proj", 48, 64, True)
            replacement = TiledNonlinAttention
        pending.append((name, module, replacement))
        counts[type(module).__name__] += 1
    scores = counts["RelPositionMultiheadAttentionWeights"]
    if not scores or counts["SelfAttention"] != 2 * scores or counts["NonlinAttention"] != scores:
        raise ValueError(f"Expected complete original score/self/nonlinear attention groups, found {counts}")

    source_path = Path(__file__).resolve().with_name("zipenhancer_tiled_attention.cu")
    helper_path = Path(__file__).resolve()
    # Resolve provenance before mutating the model, including missing source files.
    files = {"cuda_source": {"path": str(source_path), "sha256": _sha256(source_path)},
             "python_helper": {"path": str(helper_path), "sha256": _sha256(helper_path)}}
    library = _load_library(library_path)
    files["library"] = {"path": str(library.path), "sha256": library.sha256}
    paths = {kind.__name__: [] for kind in types}
    for name, original, replacement in pending:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        wrapper = replacement(original, library.op) if replacement is TiledAttentionWeights else replacement(original)
        setattr(parent, child_name, wrapper)
        paths[type(original).__name__].append(name)
    return {"experiment": "tiled_relative_attention", "custom_op": library.op_name,
            "files": files, "replaced_classes": counts, "replacement_paths": paths,
            "configuration": dict(_EXPECTED), "parameter_objects_preserved": True,
            "all_keys_preserved": True, "attention_matrices_materialized": False,
            "inference_only": True, "masks_supported": False,
            "arithmetic": {"score_dots": "FP32 accumulation rounded to FP16",
                           "score_sum": "FP16", "softmax": "online FP32",
                           "tile_probabilities": "FP16 unnormalized exponentials for WMMA",
                           "weighted_values": "FP32 accumulation, FP16 output",
                           "original_half_softmax_rounding_preserved": False},
            "relative_position_index": "N-1-query+key"}
