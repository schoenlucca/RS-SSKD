"""RS-SSKD adaptado para segmentação de copas de árvores.

Módulos:
    model:   rede do aluno (ResNet com 5 canais e CAM; será o backbone do
             Mask R-CNN).
    losses:  consistência sob transformações e auto-destilação do RS-SSKD.
    teacher: professor TEBAKA (NDVI -> máscara -> copas). Importe direto:
             from rsskd.teacher import TebakaTeacher
    inputs:  entrada do aluno (5 bandas + NDVI, normalizadas). Importe direto:
             from rsskd.inputs import InputSpec
"""

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
]
