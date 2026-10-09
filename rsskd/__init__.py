"""RS-SSKD: auto-supervisao com destilacao para classificacao few-shot."""

from .fewshot import EpisodeSampler, proto_logits
from .losses import base_loss, cam_consistency, distill_loss, make_teacher, t_flip, t_scale
from .model import RSSSKDNet

__all__ = [
    "RSSSKDNet",
    "base_loss",
    "cam_consistency",
    "distill_loss",
    "make_teacher",
    "t_flip",
    "t_scale",
    "proto_logits",
    "EpisodeSampler",
]
