"""Experimental FP16 storage with FP32 nonlinear and normalization arithmetic."""
from __future__ import annotations
import torch


class HalfBiasNorm(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        self.bias, self.log_scale = original.bias, original.log_scale
        self.channel_dim = original.channel_dim

    def forward(self, value):
        with torch.autocast('cuda', enabled=False):
            source = value.float()
            axis = self.channel_dim % source.ndim
            shape = [1] * source.ndim
            shape[axis] = self.bias.numel()
            centered = source - self.bias.float().reshape(shape)
            scale = centered.square().mean(dim=axis, keepdim=True).rsqrt() * self.log_scale.float().exp()
            return (source * scale).to(value.dtype)


class HalfSwoosh(torch.nn.Module):
    def __init__(self, offset, floor):
        super().__init__()
        self.offset, self.floor = offset, floor

    def forward(self, value):
        with torch.autocast('cuda', enabled=False):
            source = value.float()
            result = torch.logaddexp(torch.zeros((), dtype=source.dtype, device=source.device), source - self.offset) - 0.08 * source - self.floor
            return result.to(value.dtype)


def use_half_residuals(model):
    from modelscope.models.audio.ans.zipenhancer_layers.scaling import BiasNorm, SwooshL, SwooshR
    if model.training:
        raise ValueError('FP16 residual experiment requires an evaluation model')
    model.half()
    changed = []
    for name, original in list(model.named_modules()):
        replacement = None
        if isinstance(original, BiasNorm):
            replacement = HalfBiasNorm(original)
        elif isinstance(original, SwooshL):
            replacement = HalfSwoosh(4.0, 0.035)
        elif isinstance(original, SwooshR):
            replacement = HalfSwoosh(1.0, 0.313261687)
        if replacement is not None:
            parent, _, child = name.rpartition('.')
            setattr(model.get_submodule(parent) if parent else model, child, replacement.eval())
            changed.append(name)
    return {'parameterStorage': 'float16', 'residualStorage': 'preserve input dtype', 'nonlinearStatistics': 'float32', 'replacedModules': changed}
