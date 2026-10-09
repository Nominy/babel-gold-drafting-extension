"""Experimental inference-only FP32-statistics instance normalization for ZipEnhancer."""
from __future__ import annotations
import torch
import triton
import triton.language as tl


@triton.jit
def _partial(X, P, Q, N: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    index = tile * BLOCK + tl.arange(0, BLOCK)
    anchor = tl.load(X + row * N).to(tl.float32)
    value = tl.load(X + row * N + index, index < N, other=0).to(tl.float32)
    delta = tl.where(index < N, value - anchor, 0.0)
    tl.store(P + row * PARTS + tile, tl.sum(delta, 0))
    tl.store(Q + row * PARTS + tile, tl.sum(delta * delta, 0))


@triton.jit
def _statistics(X, P, Q, M, R, N: tl.constexpr, PARTS: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    index = tl.arange(0, BLOCK)
    mean_delta = tl.sum(tl.load(P + row * PARTS + index, index < PARTS, other=0), 0) / N
    square_delta = tl.sum(tl.load(Q + row * PARTS + index, index < PARTS, other=0), 0) / N
    variance = tl.maximum(square_delta - mean_delta * mean_delta, 0.0)
    # Store the anchored mean offset, not anchor+offset: this also avoids
    # cancellation for nearly constant values during the final subtraction.
    tl.store(M + row, mean_delta)
    tl.store(R + row, tl.rsqrt(variance + EPS))


@triton.jit
def _normalize(X, W, B, M, R, Y, N: tl.constexpr, C: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    index = tile * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + row * N + index, index < N, other=0).to(tl.float32)
    anchor = tl.load(X + row * N).to(tl.float32)
    mean_delta, reciprocal = tl.load(M + row), tl.load(R + row)
    scale, bias = tl.load(W + row % C), tl.load(B + row % C)
    output = ((value - anchor) - mean_delta) * reciprocal * scale + bias
    tl.store(Y + row * N + index, output, index < N)


@torch.library.custom_op('babel_zip::instance_norm', mutates_args=(), device_types='cuda')
def instance_norm(value: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    value = value.contiguous()
    n = value.shape[-1] * value.shape[-2]
    rows = value.shape[0] * value.shape[1]
    parts = triton.cdiv(n, 2048)
    partial = torch.empty((2, rows, parts), device=value.device, dtype=torch.float32)
    stats = torch.empty((2, rows), device=value.device, dtype=torch.float32)
    output = torch.empty_like(value)
    _partial[(rows, parts)](value, partial[0], partial[1], n, parts, 2048, num_warps=4)
    _statistics[(rows,)](value, partial[0], partial[1], stats[0], stats[1], n, parts, eps, triton.next_power_of_2(parts), num_warps=4)
    _normalize[(rows, triton.cdiv(n, 1024))](value, weight, bias, stats[0], stats[1], output, n, value.shape[1], 1024, num_warps=4)
    return output


@instance_norm.register_fake
def _fake(value, weight, bias, eps):
    return torch.empty_like(value, memory_format=torch.contiguous_format)


class TritonInstanceNorm(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        if not original.affine or original.track_running_stats:
            raise ValueError('Only affine, per-instance-statistics normalization is supported')
        self.weight, self.bias, self.eps = original.weight, original.bias, original.eps

    def forward(self, value):
        return instance_norm(value, self.weight, self.bias, self.eps)


def replace_instance_norm(model):
    if model.training:
        raise ValueError('Experimental normalization supports inference only')
    names = []
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.InstanceNorm2d):
            parent, _, child = name.rpartition('.')
            setattr(model.get_submodule(parent) if parent else model, child, TritonInstanceNorm(module))
            names.append(name)
    if not names:
        raise ValueError('No instance normalization layers found')
    return names
