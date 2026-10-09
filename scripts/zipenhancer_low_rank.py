"""Experimental truncated-SVD linear projections; never modifies saved weights."""
from __future__ import annotations

import math
import torch
from torch import nn


class FactorizedLinear(nn.Module):
    def __init__(self, original: nn.Module, rank: int, activation=None):
        super().__init__()
        weight = original.weight.detach()
        # Factorize once on CPU in float64; inference factors retain original dtype.
        u, s, vh = torch.linalg.svd(weight.cpu().double(), full_matrices=False)
        self.first = nn.Linear(weight.shape[1], rank, bias=False)
        self.second = nn.Linear(rank, weight.shape[0], bias=original.bias is not None)
        with torch.no_grad():
            self.first.weight.copy_(vh[:rank])
            self.second.weight.copy_(u[:, :rank] * s[:rank])
            if original.bias is not None:
                self.second.bias.copy_(original.bias.detach().cpu())
        self.to(device=weight.device, dtype=weight.dtype)
        self.activation = activation
        self.retained_energy = float(s[:rank].square().sum() / s.square().sum())
        self.eval()

    def forward(self, value):
        if self.activation is not None:
            value = self.activation(value)
        return self.second(self.first(value))


def factorize_projections(model: nn.Module, fraction: float, *, feedforward_only: bool = True) -> list[dict]:
    """Replace selected eval-only projections by actual smaller matrix products.

    The fraction is of minimum matrix dimension. Replacements which do not
    reduce scalar multiply count are skipped. No implied audio-quality gate.
    """
    if model.training or not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError('Low-rank experiment needs an eval model and rank fraction in (0, 1)')
    from modelscope.models.audio.ans.zipenhancer_layers.scaling import (
        ActivationDropoutAndLinear, SwooshLForward, SwooshRForward,
    )
    replacements = []
    for name, module in list(model.named_modules()):
        if not name or feedforward_only and 'feed_forward' not in name:
            continue
        activation = None
        if isinstance(module, ActivationDropoutAndLinear):
            if module.activation not in ('SwooshL', 'SwooshR'):
                raise ValueError(f'Unrecognized activation in {name}')
            activation = SwooshLForward if module.activation == 'SwooshL' else SwooshRForward
        elif not isinstance(module, nn.Linear):
            continue
        output_size, input_size = module.weight.shape
        rank = max(1, int(min(output_size, input_size) * fraction))
        original_products = input_size * output_size
        factor_products = rank * (input_size + output_size)
        if factor_products >= original_products:
            continue
        factor = FactorizedLinear(module, rank, activation)
        parent_name, _, child_name = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, factor)
        replacements.append({'name': name, 'shape': [output_size, input_size], 'rank': rank,
                             'retainedWeightEnergy': factor.retained_energy,
                             'arithmeticRatio': factor_products / original_products})
    if not replacements:
        raise ValueError('No eligible projection was compressed')
    return replacements
