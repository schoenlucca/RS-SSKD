"""Testes das peças do treino e da checagem de sobreajuste."""

import json

import pytest

torch = pytest.importorskip("torch")

from rsskd.train import make_optimizer, train_step, warmup_lr  # noqa: E402


class Parabola(torch.nn.Module):
    """Modelo de brinquedo: a "perda" é (w - 3)^2, mínima em w = 3."""

    def __init__(self, w=0.0):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(w))

    def forward(self, images, targets):
        return {"loss_a": (self.w - 3) ** 2, "loss_b": 0 * self.w}


def test_aquecimento_sobe_ate_a_taxa_escolhida():
    opt = make_optimizer(Parabola(), lr=0.01)
    lrs = [warmup_lr(opt, s, 10, 0.01) for s in range(15)]
    assert lrs[0] == pytest.approx(0.001)            # começa em 1/10
    assert lrs[5] == pytest.approx(0.0055)
    assert lrs[10:] == [0.01] * 5                    # depois, fica constante
    assert all(a < b for a, b in zip(lrs[:10], lrs[1:11]))
    assert opt.param_groups[0]["lr"] == 0.01


def test_passos_levam_ao_minimo():
    m = Parabola()
    opt = make_optimizer(m, lr=0.1, weight_decay=0.0)
    out = {}
    for _ in range(100):
        out = train_step(m, opt, [], [], "cpu")
    assert m.w.item() == pytest.approx(3, abs=1e-2)
    assert out["total"] < 1e-3 and not out["pulado"]


def test_passo_com_perda_infinita_e_pulado():
    m = Parabola(w=float("inf"))
    opt = make_optimizer(m, lr=0.1)
    out = train_step(m, opt, [], [], "cpu")
    assert out["pulado"] and m.w.item() == float("inf")   # pesos intactos


def test_script_de_sobreajuste_roda(tmp_path):
    # Conjunto falso de 2 imagens (o mesmo formato dos testes do carregador),
    # com uma copa do professor em cada; só confere que o script roda de
    # ponta a ponta e grava as saídas. São 2 passos, sem pré-treino.
    pytest.importorskip("matplotlib")
    import numpy as np
    pd = pytest.importorskip("pandas")
    from overfit_check import main
    from test_data import NIR, RED, H, W, _fake_row

    from rsskd.data import AgmlTreeCrowns
    from rsskd.inputs import compute_input_spec
    from rsskd.pseudolabels import PseudoLabelStore

    dados = tmp_path / "dados"
    dados.mkdir()
    rows = [_fake_row(i, "train") for i in range(2)]
    pd.DataFrame(rows).to_parquet(dados / "train-00000.parquet")
    store = PseudoLabelStore(tmp_path / "pseudo")
    for i in range(2):
        copas = np.zeros((H, W), np.int32)
        copas[10:30, 10:30] = 1
        store.save("train", i, copas)
    imgs = AgmlTreeCrowns(dados, split="train")
    spec = compute_input_spec((imgs[i].bands for i in range(2)), NIR, RED,
                              ["b0", "b1", "b2", "b3", "b4", "ndvi"])
    spec.save(tmp_path / "entrada.json")

    resumo = main(["--pasta", str(dados), "--copas", str(tmp_path / "pseudo"),
                   "--entrada", str(tmp_path / "entrada.json"), "--n-imagens", "2",
                   "--passos", "2", "--sem-pretreino", "--dispositivo", "cpu",
                   "--saida", str(tmp_path / "saida")])
    assert resumo["passos"] == 2
    for f in ["perdas.png", "copas.png", "resumo.json"]:
        assert (tmp_path / "saida" / f).exists()
    assert json.loads((tmp_path / "saida" / "resumo.json").read_text(
        encoding="utf-8"))["imagens"] == [0, 1]
