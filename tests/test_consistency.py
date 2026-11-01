"""Testes da consistência do RS-SSKD no Mask R-CNN (item 4)."""

import pytest

torch = pytest.importorskip("torch")

from rsskd.consistency import capture_objectness, objectness_maps  # noqa: E402
from rsskd.detector import build_student  # noqa: E402

S = 64


@pytest.fixture(scope="module")
def aluno():
    torch.manual_seed(0)
    m = build_student(pretrained_backbone=False, image_size=S)
    m.eval()                   # BatchNorm determinística nos testes
    return m


def _alvo():
    masks = torch.zeros(1, S, S, dtype=torch.uint8)
    masks[0, 10:30, 10:30] = 1
    return {"boxes": torch.tensor([[10.0, 10, 30, 30]]),
            "labels": torch.ones(1, dtype=torch.int64), "masks": masks}


def test_formato_dos_mapas(aluno):
    x = torch.randn(2, 6, S, S)
    mapas = objectness_maps(aluno, x)
    # 5 níveis da FPN, 3 âncoras por posição, passos 4, 8, 16, 32 e 64.
    assert [tuple(m.shape) for m in mapas] == [
        (2, 3, 16, 16), (2, 3, 8, 8), (2, 3, 4, 4), (2, 3, 2, 2), (2, 3, 1, 1)]


def test_escala_gera_mapas_maiores(aluno):
    # Uma imagem ampliada não é redimensionada de volta: os mapas crescem.
    mapas = objectness_maps(aluno, torch.randn(1, 6, 96, 96))
    assert tuple(mapas[0].shape[-2:]) == (24, 24)


def test_captura_no_treino_igual_ao_calculo_direto(aluno):
    # O gancho guarda os mesmos mapas que o cálculo direto produziria, e com
    # gradiente (para poderem entrar numa perda).
    x = torch.randn(6, S, S)
    aluno.train()
    for mod in aluno.modules():                   # BatchNorm sem estatística
        if isinstance(mod, torch.nn.BatchNorm2d):  # do lote, para comparar
            mod.eval()
    with capture_objectness(aluno) as cap:
        aluno([x], [_alvo()])
    direto = objectness_maps(aluno, x[None])
    assert len(cap["objectness"]) == 5
    for a, b in zip(cap["objectness"], direto):
        assert torch.allclose(a, b, atol=1e-5)
    assert cap["objectness"][0].requires_grad
    aluno.eval()


def test_gancho_e_removido(aluno):
    with capture_objectness(aluno) as cap:
        pass
    with torch.no_grad():
        aluno([torch.randn(6, S, S)])
    assert "objectness" not in cap               # fora do 'with', não escuta
