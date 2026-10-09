"""Opt-in native inference experiments; original checkpoint files stay untouched."""
from __future__ import annotations
import torch


def install_fused_attention_softmax():
    from modelscope.models.audio.ans.zipenhancer_layers import zipformer
    original = zipformer.softmax

    def fused_softmax(value, dim):
        # CUDA softmax accumulates half inputs in FP32 internally. Avoid the
        # reference's separate max/subtract/exp/sum/divide tensor passes.
        with torch.autocast('cuda', enabled=False):
            return torch.softmax(value, dim=dim)

    zipformer.softmax = fused_softmax
    return original


def restore_attention_softmax(original):
    from modelscope.models.audio.ans.zipenhancer_layers import zipformer
    zipformer.softmax = original
