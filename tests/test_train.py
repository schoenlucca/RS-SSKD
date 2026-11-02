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


@pytest.fixture()
def dados_falsos(tmp_path):
    """Conjunto falso pequeno (2 imagens de treino e 2 de validação, no
    formato dos testes do carregador), com uma copa do professor em cada e o
    arquivo de normalização. Devolve (pasta dos dados, pasta das copas,
    arquivo de entrada)."""
    import numpy as np
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from test_data import NIR, RED, H, W, _fake_row

    from rsskd.data import AgmlTreeCrowns
    from rsskd.inputs import compute_input_spec
    from rsskd.pseudolabels import PseudoLabelStore

    dados = tmp_path / "dados"
    dados.mkdir()
    rows = [_fake_row(i, s) for i, s in enumerate(["train", "train", "val", "val"])]
    pd.DataFrame(rows).to_parquet(dados / "train-00000.parquet")
    store = PseudoLabelStore(tmp_path / "pseudo")
    for split in ("train", "val"):
        for i in range(2):
            copas = np.zeros((H, W), np.int32)
            copas[10:30, 10:30] = 1
            store.save(split, i, copas)
    imgs = AgmlTreeCrowns(dados, split="train")
    spec = compute_input_spec((imgs[i].bands for i in range(2)), NIR, RED,
                              ["b0", "b1", "b2", "b3", "b4", "ndvi"])
    spec.save(tmp_path / "entrada.json")
    return dados, tmp_path / "pseudo", tmp_path / "entrada.json"


def test_script_de_sobreajuste_roda(dados_falsos, tmp_path):
    # Só confere que o script roda de ponta a ponta e grava as saídas:
    # 2 passos, sem pré-treino.
    pytest.importorskip("matplotlib")
    from overfit_check import main

    dados, pseudo, entrada = dados_falsos
    resumo = main(["--pasta", str(dados), "--copas", str(pseudo),
                   "--entrada", str(entrada), "--n-imagens", "2",
                   "--passos", "2", "--sem-pretreino", "--dispositivo", "cpu",
                   "--saida", str(tmp_path / "saida")])
    assert resumo["passos"] == 2
    for f in ["perdas.png", "copas.png", "resumo.json"]:
        assert (tmp_path / "saida" / f).exists()
    assert json.loads((tmp_path / "saida" / "resumo.json").read_text(
        encoding="utf-8"))["imagens"] == [0, 1]


def test_passo_com_consistencia_do_rsskd():
    # Com o aluno de verdade (pequeno, sem pré-treino): a consistência entra
    # na perda total com o peso lambda e aparece separada por transformação.
    from rsskd.detector import build_student

    torch.manual_seed(0)
    S = 64
    m = build_student(pretrained_backbone=False, image_size=S)
    masks = torch.zeros(1, S, S, dtype=torch.uint8)
    masks[0, 10:30, 10:30] = 1
    alvo = {"boxes": torch.tensor([[10.0, 10, 30, 30]]),
            "labels": torch.ones(1, dtype=torch.int64), "masks": masks}
    opt = make_optimizer(m, lr=0.0)                      # lr 0: pesos não mudam
    imgs = [torch.randn(6, S, S), torch.randn(6, S, S)]
    sem = train_step(m, opt, imgs, [alvo, alvo], "cpu")
    com = train_step(m, opt, imgs, [alvo, alvo], "cpu", consistency="artigo",
                     consistency_weight=5.0)
    assert "loss_consistencia" not in sem
    assert {"cons_flip", "cons_scale", "cons_flip+scale"} <= set(com)
    assert com["loss_consistencia"] == pytest.approx(
        com["cons_flip"] + com["cons_scale"] + com["cons_flip+scale"])
    mrcnn = sum(com[k] for k in com if k.startswith("loss_") and k != "loss_consistencia")
    assert com["total"] == pytest.approx(mrcnn + 5.0 * com["loss_consistencia"], rel=1e-4)


def test_taxa_aquecimento_e_cosseno():
    from rsskd.train import lr_schedule

    opt = make_optimizer(Parabola(), lr=0.01)
    lrs = [lr_schedule(opt, s, 110, 10, 0.01) for s in range(111)]
    assert lrs[0] == pytest.approx(0.001)                 # aquecimento
    assert lrs[10] == pytest.approx(0.01)                 # topo
    assert lrs[60] == pytest.approx(0.01 * (0.01 + 0.99 * 0.5))   # meio
    assert lrs[110] == pytest.approx(0.0001)              # fim: 1% da taxa
    assert all(a >= b for a, b in zip(lrs[10:], lrs[11:]))   # só desce


def test_script_de_treino_e_retomada(dados_falsos, tmp_path):
    import csv

    from train_student import main

    dados, pseudo, entrada = dados_falsos
    base = ["--nome", "teste", "--pasta", str(dados), "--copas", str(pseudo),
            "--entrada", str(entrada), "--lote", "1", "--sem-pretreino",
            "--dispositivo", "cpu", "--saida", str(tmp_path / "runs"),
            "--aquecimento", "1", "--max-passos-por-epoca", "1"]
    run = main(base + ["--epocas", "2"])
    linhas = list(csv.DictReader(open(run / "historico.csv", encoding="utf-8")))
    assert [r["epoca"] for r in linhas] == ["0", "1"]
    assert "loss_consistencia" in linhas[0]               # padrão: artigo
    assert (run / "ultimo.pt").exists() and (run / "config.json").exists()

    # Retomar até a época 3: só a época 2 é treinada, e o histórico continua.
    main(base + ["--epocas", "3", "--continuar"])
    linhas = list(csv.DictReader(open(run / "historico.csv", encoding="utf-8")))
    assert [r["epoca"] for r in linhas] == ["0", "1", "2"]
    ckpt = torch.load(run / "ultimo.pt", weights_only=False)
    assert ckpt["epoca"] == 2 and ckpt["passo"] == 3


def test_lambda_zero_desliga_a_consistencia(dados_falsos, tmp_path):
    import csv

    from train_student import main

    dados, pseudo, entrada = dados_falsos
    run = main(["--nome", "l0", "--pasta", str(dados), "--copas", str(pseudo),
                "--entrada", str(entrada), "--lote", "1", "--sem-pretreino",
                "--dispositivo", "cpu", "--saida", str(tmp_path / "runs"),
                "--epocas", "1", "--max-passos-por-epoca", "1",
                "--peso-consistencia", "0"])
    linha = next(csv.DictReader(open(run / "historico.csv", encoding="utf-8")))
    assert "loss_consistencia" not in linha
