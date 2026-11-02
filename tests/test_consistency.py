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


# --------------------------------------------------------------------------
# Parte 4b: transformações e perda
# --------------------------------------------------------------------------
from collections import OrderedDict  # noqa: E402

import torch.nn.functional as F  # noqa: E402

from rsskd.consistency import (  # noqa: E402
    Transform, choose_transforms, consistency_loss,
)


class Equivariante(torch.nn.Module):
    """Rede de mentira que respeita exatamente espelhamentos e rotações.

    O "backbone" só faz médias em blocos 2x2 (que comutam com espelhar e
    girar imagens de lado par), e a "RPN" é uma convolução 1x1 (pixel a
    pixel). As âncoras 0 e 2 usam o mesmo filtro, para que a troca de canais
    das rotações também dê o mesmo resultado. Numa rede assim, a perda de
    consistência de espelhamentos e rotações tem de ser zero.
    """

    def __init__(self, vies=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 2, 1)
        self.vies = vies          # None, "rampa" ou "xadrez": quebram a coerência
        outer = self

        class Head(torch.nn.Module):
            def forward(self, feats):
                outs = []
                for f in feats:
                    a, b = outer.conv(f).unbind(1)
                    o = torch.stack([a, b, a], 1)
                    h, w = o.shape[-2:]
                    if outer.vies == "rampa":      # depende da altura na imagem
                        o = o + torch.linspace(-3, 3, h)[:, None].expand(h, w)
                    elif outer.vies == "xadrez":   # depende da posição exata
                        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w),
                                                indexing="ij")
                        o = o + 3.0 * ((yy + xx) % 2)
                    outs.append(o)
                return outs, None

        class Rpn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = Head()

        self.rpn = Rpn()

    def backbone(self, x):
        return OrderedDict((str(i), F.avg_pool2d(x, 2 ** (i + 1))) for i in range(3))


def _consistencia(rede, nomes, suave=False):
    x = torch.randn(2, 6, 32, 32)
    if suave:   # imagem sem detalhes finos: a interpolação quase não a altera
        x = F.interpolate(torch.randn(2, 6, 4, 4), size=(32, 32), mode="bilinear")
    orig = objectness_maps(rede, x)
    return consistency_loss(rede, x, orig, [Transform(n) for n in nomes])


def test_rede_coerente_tem_perda_zero_em_espelhos_e_rotacoes():
    torch.manual_seed(0)
    nomes = ["flip", "hflip", "rot90", "rot180", "rot270", "rot90+hflip"]
    total, partes = _consistencia(Equivariante(), nomes)
    assert set(partes) == set(nomes)
    assert total.item() < 1e-10


def test_rede_incoerente_tem_perda_positiva():
    torch.manual_seed(0)
    total, _ = _consistencia(Equivariante(vies="rampa"), ["flip"])
    assert total.item() > 1e-2


def test_escala_quase_coerente():
    # A escala usa interpolação, então nem uma rede coerente fica com perda
    # exatamente zero; mas fica bem menor que a de uma rede incoerente.
    torch.manual_seed(0)
    boa, _ = _consistencia(Equivariante(), ["scale"], suave=True)
    ruim, _ = _consistencia(Equivariante(vies="xadrez"), ["scale"], suave=True)
    assert boa.item() < 0.1 * ruim.item()


def test_escolha_das_transformacoes():
    assert [t.name for t in choose_transforms("artigo")] == ["flip", "scale", "flip+scale"]
    g = torch.Generator().manual_seed(0)
    sorteadas = {choose_transforms("sorteio", g)[0].name for _ in range(200)}
    assert len(sorteadas) == 7                    # todas menos a identidade
    assert [t.name for t in choose_transforms("flip, rot90")] == ["flip", "rot90"]
    with pytest.raises(ValueError):
        Transform("girar")


def test_imagem_escalada_fica_maior():
    x = torch.randn(1, 6, 256, 256)
    assert Transform("scale").image(x).shape[-1] == 307          # 256 x 1,2
    assert torch.equal(Transform("flip").image(x), x.flip(-2))


def test_perda_treina_o_aluno(aluno):
    # Com o aluno de verdade: a perda é finita e gera gradiente no backbone.
    aluno.train()
    x = torch.randn(1, 6, S, S)
    with capture_objectness(aluno) as cap:
        aluno(list(x), [_alvo()])
    total, partes = consistency_loss(aluno, x, cap["objectness"],
                                     choose_transforms("artigo"))
    assert torch.isfinite(total) and set(partes) == {"flip", "scale", "flip+scale"}
    aluno.zero_grad()
    total.backward()
    assert aluno.backbone.body.layer4[0].conv1.weight.grad is not None
    aluno.eval()


def test_rotacao_troca_as_ancoras_largas_e_altas():
    # Canais = proporções 0,5 (larga), 1 e 2 (alta). Girando 90°, o que era
    # largo vira alto: o canal 0 vai para o 2 e vice-versa; o 1 fica.
    m = torch.randn(1, 3, 8, 8)
    r = Transform("rot90").maps(m, (8, 8))
    assert torch.equal(r[:, 0], torch.rot90(m[:, 2], 1, dims=(-2, -1)))
    assert torch.equal(r[:, 2], torch.rot90(m[:, 0], 1, dims=(-2, -1)))
    assert torch.equal(r[:, 1], torch.rot90(m[:, 1], 1, dims=(-2, -1)))
    # Espelhar e girar 180° não trocam as proporções.
    assert torch.equal(Transform("flip").maps(m, (8, 8)), m.flip(-2))
    assert torch.equal(Transform("rot180").maps(m, (8, 8)),
                       torch.rot90(m, 2, dims=(-2, -1)))
