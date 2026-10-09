"""Teste rapido com dados sinteticos. Rodar com: pytest

Usa ResNet-18 nas perdas para caber em CPU; a ResNet-101 e conferida so
nas dimensoes de saida.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsskd import (  # noqa: E402
    EpisodeSampler, RSSSKDNet, base_loss, distill_loss, make_teacher, proto_logits,
)

N_CLASSES, CHANNELS = 6, 5


@pytest.fixture(scope="module")
def small():
    torch.manual_seed(0)
    return RSSSKDNet(N_CLASSES, in_channels=CHANNELS, backbone="resnet18", pretrained=False)


@pytest.fixture(scope="module")
def batch():
    torch.manual_seed(0)
    return torch.randn(2, CHANNELS, 96, 96), torch.randint(0, N_CLASSES, (2,))


def test_cam_reproduz_logits(small, batch):
    x, _ = batch
    logits, emb, cam = small(x, return_cam=True)
    assert logits.shape == (2, N_CLASSES)
    assert emb.shape == (2, small.embed_dim)
    assert cam.shape[:2] == (2, N_CLASSES)
    # A media espacial do CAM mais o bias da FC tem de dar os logits.
    recon = cam.mean(dim=(2, 3)) + small.fc.bias
    assert torch.allclose(recon, logits, atol=1e-4)


def test_base_loss_tem_gradiente(small, batch):
    x, y = batch
    small.zero_grad()
    out = base_loss(small, x, y)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    assert small.fc.weight.grad.abs().sum() > 0


def test_distill_loss(small, batch):
    x, y = batch
    teacher = make_teacher(small)
    assert not any(p.requires_grad for p in teacher.parameters())
    small.zero_grad()
    out = distill_loss(small, teacher, x, y)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])


def test_resnet101_dimensoes():
    model = RSSSKDNet(N_CLASSES, in_channels=CHANNELS, pretrained=False).eval()
    assert model.embed_dim == 2048
    sizes = {}
    with torch.no_grad():
        for s in (80, 96, 224):
            _, _, cam = model(torch.randn(1, CHANNELS, s, s), return_cam=True)
            sizes[s] = tuple(cam.shape[-2:])
    # Reducao de 32x: 80 e 96 dao o mesmo CAM, o que esvazia a perda de escala.
    assert sizes == {80: (3, 3), 96: (3, 3), 224: (7, 7)}


def test_episodios_e_prototipos():
    labels = [c for c in range(N_CLASSES) for _ in range(12)]
    sampler = EpisodeSampler(labels, n_way=5, k_shot=5, n_query=3, n_episodes=3, seed=0)
    for sup, qry, classes in sampler:
        assert len(sup) == 25 and len(qry) == 15 and len(set(classes)) == 5
        assert not set(sup) & set(qry)

    # Consultas iguais aos prototipos tem de ser classificadas corretamente.
    protos = torch.eye(5, 16)
    sy = torch.arange(5).repeat_interleave(5)
    scores = proto_logits(protos.repeat_interleave(5, 0), sy, protos, n_way=5)
    assert scores.argmax(1).tolist() == [0, 1, 2, 3, 4]
    assert torch.allclose(scores.max(1).values, torch.full((5,), 10.0))
