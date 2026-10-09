"""Perdas do RS-SSKD: consistencia de CAM e auto-destilacao."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def t_flip(x: torch.Tensor) -> torch.Tensor:
    """Flip vertical (vale para imagens e para CAMs)."""
    return torch.flip(x, dims=[2])


def t_scale(x: torch.Tensor, factor: float = 1.2) -> torch.Tensor:
    """Escala por interpolacao bilinear (no artigo: 80x80 -> 96x96, fator 1.2)."""
    return F.interpolate(x, scale_factor=factor, mode="bilinear", align_corners=False)


def cam_consistency(cam_orig: torch.Tensor, cam_trans: torch.Tensor, flip: bool) -> torch.Tensor:
    """Eq. (2): MSE entre T(CAM(x)) e CAM(T(x)).

    A escala e aplicada redimensionando o CAM original para o tamanho do CAM
    da imagem transformada; o flip e aplicado diretamente.
    """
    target = t_flip(cam_orig) if flip else cam_orig
    if target.shape[-2:] != cam_trans.shape[-2:]:
        target = F.interpolate(
            target, size=cam_trans.shape[-2:], mode="bilinear", align_corners=False
        )
    return F.mse_loss(target, cam_trans)


def base_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    scale_factor: float = 1.2,
    ss_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Eq. (3) e (4): L_base = L_ce + L_ss(flip) + L_ss(escala) + L_ss(flip+escala)."""
    logits, _, cam = model(x, return_cam=True)
    ce = F.cross_entropy(logits, y)

    x_f = t_flip(x)
    x_s = t_scale(x, scale_factor)
    x_fs = t_scale(x_f, scale_factor)

    _, _, cam_f = model(x_f, return_cam=True)
    _, _, cam_s = model(x_s, return_cam=True)
    _, _, cam_fs = model(x_fs, return_cam=True)

    ss_f = cam_consistency(cam, cam_f, flip=True)
    ss_s = cam_consistency(cam, cam_s, flip=False)
    ss_fs = cam_consistency(cam, cam_fs, flip=True)
    ss = ss_f + ss_s + ss_fs

    return {
        "loss": ce + ss_weight * ss,
        "ce": ce.detach(),
        "ss_flip": ss_f.detach(),
        "ss_scale": ss_s.detach(),
        "ss_flip_scale": ss_fs.detach(),
        "acc": (logits.argmax(1) == y).float().mean(),
    }


def distill_loss(
    student: nn.Module,
    teacher: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    temperature: float = 4.0,
    kd_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Eq. (6): L = L_ce(aluno, rotulos) + KL(aluno || professor).

    O professor e uma copia congelada do modelo treinado com base_loss.
    A temperatura nao aparece no trecho lido do artigo: 4.0 e uma escolha
    inicial e deve ser tratada como hiperparametro.
    """
    with torch.no_grad():
        t_logits, _ = teacher(x)
    s_logits, _ = student(x)

    ce = F.cross_entropy(s_logits, y)
    kd = F.kl_div(
        F.log_softmax(s_logits / temperature, dim=1),
        F.softmax(t_logits / temperature, dim=1),
        reduction="batchmean",
    ) * temperature**2

    return {"loss": ce + kd_weight * kd, "ce": ce.detach(), "kd": kd.detach()}


def make_teacher(model: nn.Module) -> nn.Module:
    """Copia congelada do modelo, para servir de professor."""
    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher
