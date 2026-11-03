"""Testes da regra 1 de Harmon et al. (área média), etapa 2 (parte 6b)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rsskd.rules import (  # noqa: E402
    bbox_size, box_areas, reference_area, rule_loss, rule_teacher, rules_loss_from_capture,
)

MU = 1000.0


def test_sigmoide_de_area():
    a = torch.tensor([200.0, 1000.0, 1500.0, 3000.0])
    s = bbox_size(a, MU, k=5.0)
    assert s[1].item() == pytest.approx(0.5)          # na referência: 0,5
    assert s[0] > 0.95 and s[2] < 0.1 and s[3] < 1e-3  # pequena ~1, grande ~0
    assert all(s[i] > s[i + 1] for i in range(3))      # decresce com a área


def test_professor_das_regras_rebaixa_so_as_copas_grandes():
    # Três regiões com a mesma probabilidade de copa (80%), de áreas
    # pequena, média e grande.
    logits = torch.log(torch.tensor([[0.2, 0.8]] * 3))
    areas = torch.tensor([300.0, 1000.0, 3000.0])
    q = rule_teacher(logits, areas, MU, k=5.0, C=1.0, lam=2.0)
    assert torch.allclose(q.sum(1), torch.ones(3))
    assert q[0, 1] > 0.8                    # pequena: a regra reforça "copa"
    assert q[2, 1] < 0.5                    # grande: a regra derruba "copa"
    assert q[0, 1] > q[1, 1] > q[2, 1]
    # Sem confiança na regra (lambda = 0), o professor é o próprio aluno.
    q0 = rule_teacher(logits, areas, MU, lam=0.0)
    assert torch.allclose(q0, torch.softmax(logits, 1))


def test_perda_da_regra_empurra_o_aluno():
    logits = torch.log(torch.tensor([[0.2, 0.8], [0.2, 0.8]])).requires_grad_(True)
    areas = torch.tensor([300.0, 3000.0])
    loss = rule_loss(logits, areas, MU, lam=2.0)
    loss.backward()
    assert loss.item() > 0
    g = logits.grad
    # Descer o gradiente aumenta a nota de copa da pequena e diminui a da grande.
    assert g[0, 1] < 0 and g[1, 1] > 0
    assert rule_loss(torch.zeros(0, 2), torch.zeros(0), MU).item() == 0


def test_regra_nas_regioes_capturadas():
    cap = {"class_logits": torch.tensor([[0.0, 2.0], [0.0, 2.0], [2.0, 0.0]]),
           "proposals": [torch.tensor([[0.0, 0, 10, 10], [0, 0, 60, 60]]),
                         torch.tensor([[0.0, 0, 20, 20]])]}
    assert box_areas(cap["proposals"]).tolist() == [100, 3600, 400]
    loss, partes = rules_loss_from_capture(cap, MU)
    assert loss.item() > 0
    # Regiões tidas como copa: 100 px (satisfaz) e 3600 px (viola).
    s = bbox_size(torch.tensor([100.0, 3600.0]), MU).mean().item()
    assert partes["regra1_satisfeita"] == pytest.approx(s)


def test_area_de_referencia_das_copas():
    a = np.zeros((50, 50), np.int32)
    a[0:10, 0:10] = 1                 # caixa 100
    a[20:40, 20:30] = 2               # caixa 200
    b = np.zeros((50, 50), np.int32)
    b[0:30, 0:30] = 1                 # caixa 900
    assert reference_area([a, b]) == 200                    # mediana
    assert reference_area([a, b], stat="media") == pytest.approx(400)
    with pytest.raises(ValueError):
        reference_area([np.zeros((5, 5), np.int32)])
