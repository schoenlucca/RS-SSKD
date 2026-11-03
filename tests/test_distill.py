"""Testes da auto-destilação do RS-SSKD no Mask R-CNN (etapa 2, parte 6a)."""

import pytest

torch = pytest.importorskip("torch")

from rsskd.detector import build_student  # noqa: E402
from rsskd.distill import (  # noqa: E402
    capture_roi, distillation_loss, freeze, kd_classification, kd_masks,
)

S = 64


def _alvo():
    masks = torch.zeros(2, S, S, dtype=torch.uint8)
    masks[0, 5:25, 5:25] = 1
    masks[1, 35:60, 30:55] = 1
    return {"boxes": torch.tensor([[5.0, 5, 25, 25], [30, 35, 55, 60]]),
            "labels": torch.ones(2, dtype=torch.int64), "masks": masks}


def _aluno(seed):
    torch.manual_seed(seed)
    m = build_student(pretrained_backbone=False, image_size=S)
    m.train()
    # Sem pré-treino, a BatchNorm é a comum; no modo de avaliação ela usa
    # estatísticas fixas, para aluno e professor calcularem igual.
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.eval()
    return m


def _passo(aluno, professor):
    torch.manual_seed(1)
    imgs = [torch.randn(6, S, S), torch.randn(6, S, S)]
    with capture_roi(aluno) as cap:
        perdas = aluno(imgs, [_alvo(), _alvo()])
    total, partes = distillation_loss(cap, professor)
    return perdas, total, partes, cap


def test_kl_zero_quando_iguais_e_positiva_quando_diferentes():
    a = torch.randn(10, 2)
    assert kd_classification(a, a, 4.0).item() == pytest.approx(0, abs=1e-6)
    assert kd_classification(a, a + torch.randn(10, 2), 4.0).item() > 0
    m = torch.randn(5, 2, 28, 28)
    assert kd_masks(m, m, 4.0).item() == pytest.approx(0, abs=1e-5)
    assert kd_masks(m, m + torch.randn(5, 2, 28, 28), 4.0).item() > 0
    # Sem regiões, a perda é zero (e não um erro).
    assert kd_masks(torch.zeros(0, 2, 28, 28), torch.zeros(0, 2, 28, 28), 4.0).item() == 0


def test_aluno_igual_ao_professor_tem_destilacao_zero():
    aluno = _aluno(0)
    professor = freeze(aluno)
    _, total, partes, cap = _passo(aluno, professor)
    assert cap["class_logits"].shape[1] == 2                   # fundo, copa
    assert cap["mask_logits"].shape[-2:] == (28, 28)
    assert partes["kd_cls"] == pytest.approx(0, abs=1e-5)
    assert partes["kd_mask"] == pytest.approx(0, abs=1e-5)


def test_aluno_diferente_aprende_e_professor_fica_congelado():
    professor = freeze(_aluno(0))
    aluno = _aluno(1)                                # outros pesos iniciais
    antes = [p.clone() for p in professor.parameters()]
    _, total, partes, _ = _passo(aluno, professor)
    assert partes["kd_cls"] > 0 and partes["kd_mask"] > 0
    total.backward()
    # O gradiente chega às cabeças do aluno...
    cabecas = dict(aluno.named_parameters())
    assert cabecas["roi_heads.box_predictor.cls_score.weight"].grad is not None
    assert cabecas["roi_heads.mask_predictor.mask_fcn_logits.weight"].grad is not None
    # ...e o professor não tem gradiente nem muda.
    assert all(p.grad is None and not p.requires_grad for p in professor.parameters())
    assert all(torch.equal(a, b) for a, b in zip(antes, professor.parameters()))


def test_imagem_sem_copas():
    professor = freeze(_aluno(0))
    aluno = _aluno(1)
    vazio = {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64),
             "masks": torch.zeros(0, S, S, dtype=torch.uint8)}
    with capture_roi(aluno) as cap:
        aluno([torch.randn(6, S, S)], [vazio])
    total, partes = distillation_loss(cap, professor)
    assert torch.isfinite(total) and partes["kd_mask"] == 0   # nenhuma região positiva
