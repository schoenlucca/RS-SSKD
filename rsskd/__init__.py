"""RS-SSKD adaptado para segmentação de copas de árvores.

Módulos:
    model:   rede do aluno (ResNet com 5 canais e CAM; será o backbone do
             Mask R-CNN).
    losses:  consistência sob transformações e auto-destilação do RS-SSKD.
    teacher: professor TEBAKA (NDVI -> máscara -> copas). Importe direto:
             from rsskd.teacher import TebakaTeacher
    inputs:  entrada do aluno (5 bandas + NDVI, normalizadas). Importe direto:
             from rsskd.inputs import InputSpec
    dataset: imagens + copas do professor no formato do Mask R-CNN:
             from rsskd.dataset import CrownDataset
    augment: rotações/espelhamentos (aumento e consistência) e lotes:
             from rsskd.augment import RandomDihedral, make_loader
    detector: o aluno, Mask R-CNN com ResNet-101 + FPN e 6 canais:
             from rsskd.detector import build_student
    evaluation: previsões -> copas e métricas (professor e aluno):
             from rsskd.evaluation import evaluate_student, CrownMetrics
    train:   passo de treino, aquecimento da taxa e otimizador:
             from rsskd.train import train_step, warmup_lr, make_optimizer
    consistency: consistência do RS-SSKD no Mask R-CNN (mapas da RPN):
             from rsskd.consistency import objectness_maps
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
