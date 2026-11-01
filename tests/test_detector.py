"""Testes do aluno (Mask R-CNN com ResNet-101 e 6 canais).

Usam a rede sem pré-treino (nada é baixado) e imagens pequenas, para rodar
em CPU. Cada teste leva alguns segundos por causa do tamanho da ResNet-101.
"""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from rsskd.detector import adapt_first_conv, build_student, count_parameters  # noqa: E402

S = 128   # lado das imagens de teste


@pytest.fixture(scope="module")
def aluno():
    torch.manual_seed(0)
    return build_student(pretrained_backbone=False, image_size=S)


def _alvo():
    masks = torch.zeros(2, S, S, dtype=torch.uint8)
    masks[0, 10:40, 10:40] = 1
    masks[1, 60:100, 70:110] = 1
    boxes = torch.tensor([[10, 10, 40, 40], [70, 60, 110, 100]], dtype=torch.float32)
    return {"boxes": boxes, "labels": torch.ones(2, dtype=torch.int64), "masks": masks}


def test_primeira_convolucao_com_6_canais():
    rgb = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
    new = adapt_first_conv(rgb, 6, pretrained=True)
    assert new.weight.shape == (64, 6, 7, 7)
    w = rgb.weight.detach()
    assert torch.allclose(new.weight[:, :3], w * 0.5)                     # RGB x 3/6
    assert torch.allclose(new.weight[:, 4], w.mean(1) * 0.5)              # extra
    # Numa imagem em que os 6 canais são iguais (tons de cinza), a nova
    # convolução dá exatamente a saída da original com 3 canais: a escala
    # 3/6 mantém a ordem de grandeza do pré-treino.
    cinza = torch.randn(1, 1, 16, 16)
    with torch.no_grad():
        assert torch.allclose(new(cinza.repeat(1, 6, 1, 1)),
                              rgb(cinza.repeat(1, 3, 1, 1)), atol=1e-5)


def test_treino_devolve_as_perdas(aluno):
    aluno.train()
    perdas = aluno([torch.randn(6, S, S), torch.randn(6, S, S)], [_alvo(), _alvo()])
    # As 5 perdas do Mask R-CNN: classificação e caixa (RPN e cabeça),
    # e a perda das máscaras.
    assert set(perdas) == {"loss_objectness", "loss_rpn_box_reg", "loss_classifier",
                           "loss_box_reg", "loss_mask"}
    total = torch.stack(list(perdas.values())).sum()
    total.backward()
    assert aluno.backbone.body.conv1.weight.grad is not None   # 1a conv treina


def test_imagem_sem_copas_no_treino(aluno):
    aluno.train()
    vazio = {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64),
             "masks": torch.zeros(0, S, S, dtype=torch.uint8)}
    perdas = aluno([torch.randn(6, S, S)], [vazio])
    assert all(torch.isfinite(v) for v in perdas.values())


def test_previsao(aluno):
    aluno.eval()
    with torch.no_grad():
        (out,) = aluno([torch.randn(6, S, S)])
    n = len(out["scores"])
    assert out["boxes"].shape == (n, 4)
    assert out["masks"].shape == (n, 1, S, S)        # máscaras no tamanho da imagem
    assert set(out["labels"].tolist()) <= {1}


def test_camadas_congeladas(aluno):
    body = aluno.backbone.body
    assert body.conv1.weight.requires_grad                      # treina
    assert not any(p.requires_grad for p in body.layer1.parameters())   # congelada
    assert all(p.requires_grad for p in body.layer4.parameters())       # treina
    n = count_parameters(aluno)
    assert 50e6 < n["total"] < 70e6 and n["treinaveis"] < n["total"]
