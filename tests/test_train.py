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
    run = main(base + ["--epocas", "2", "--paciencia", "0"])
    linhas = list(csv.DictReader(open(run / "historico.csv", encoding="utf-8")))
    assert [r["epoca"] for r in linhas] == ["0", "1"]
    assert "loss_consistencia" in linhas[0]               # padrão: artigo
    assert {"val_f1", "valprof_f1", "melhor"} <= set(linhas[0])   # validação
    for f in ["ultimo.pt", "melhor.pt", "config.json"]:
        assert (run / f).exists()

    # Retomar até a época 3: só a época 2 é treinada, e o histórico continua.
    main(base + ["--epocas", "3", "--continuar", "--paciencia", "0"])
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


def test_parada_antecipada_e_avaliacao(dados_falsos, tmp_path):
    import csv

    from evaluate_student import main as avaliar
    from train_student import main as treinar

    dados, pseudo, entrada = dados_falsos
    # Um aluno quase sem treino não melhora o F1 depois da 1a época: com
    # paciência 1, o treino tem de parar na época 1, e não ir até a 5.
    run = treinar(["--nome", "parada", "--pasta", str(dados), "--copas", str(pseudo),
                   "--entrada", str(entrada), "--lote", "1", "--sem-pretreino",
                   "--dispositivo", "cpu", "--saida", str(tmp_path / "runs"),
                   "--epocas", "5", "--max-passos-por-epoca", "1",
                   "--lr", "0", "--paciencia", "1"])
    linhas = list(csv.DictReader(open(run / "historico.csv", encoding="utf-8")))
    assert [r["epoca"] for r in linhas] == ["0", "1"]
    assert linhas[0]["melhor"] == "1" and linhas[1]["melhor"] == "0"
    assert torch.load(run / "melhor.pt", weights_only=False)["epoca"] == 0

    # O script de avaliação carrega o melhor modelo e grava as saídas.
    s_ref, s_prof = avaliar(["--run", str(run), "--dispositivo", "cpu", "--n-figura", "2"])
    assert set(s_ref) == set(s_prof) and "f1" in s_ref
    for ext in ["json", "csv", "png"]:
        assert (run / f"avaliacao_val_melhor.{ext}").exists()


def test_comparacao_dos_treinos(tmp_path):
    # Dois treinos falsos (só config e histórico) e um antigo, sem validação.
    pytest.importorskip("matplotlib")
    import csv

    from compare_runs import main as comparar

    def treino(nome, lam, linhas):
        d = tmp_path / "runs" / nome
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps(
            {"peso_consistencia": lam, "consistencia": "artigo", "criterio": "professor"}),
            encoding="utf-8")
        with open(d / "historico.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(linhas[0]))
            w.writeheader()
            w.writerows(linhas)

    def linha(ep, f1, prof, melhor):
        return {"epoca": ep, "tempo_s": 60, "loss_mask": 0.3, "val_f1": f1,
                "val_precisao": f1, "val_revocacao": f1, "val_iou_copa": 0.7,
                "val_erro_contagem": 2.0, "valprof_f1": prof, "melhor": melhor}

    treino("lambda0", 0, [linha(0, 0.40, 0.60, 1), linha(1, 0.45, 0.70, 1),
                          linha(2, 0.50, 0.65, 0)])
    treino("lambda100", 100, [linha(0, 0.42, 0.62, 1), linha(1, 0.52, 0.61, 0)])
    treino("antigo", 1, [{"epoca": 0, "tempo_s": 60, "loss_mask": 0.4}])

    linhas = comparar(["--pasta-runs", str(tmp_path / "runs"), "--pasta",
                       str(tmp_path / "nada"), "--saida", str(tmp_path / "saidas")])
    por_nome = {r["treino"]: r for r in linhas}
    # A época escolhida é a do melhor.pt (a última marcada como melhor),
    # mesmo que outra época tenha F1 maior contra a referência.
    assert por_nome["lambda0"]["epoca_escolhida"] == 1
    assert por_nome["lambda0"]["ref_f1"] == 0.45
    assert por_nome["lambda100"]["consistencia"] == "artigo"
    assert por_nome["lambda0"]["consistencia"] == "desligada"
    assert por_nome["antigo"]["ref_f1"] == ""          # sem validação: em branco
    assert por_nome["lambda0"]["tempo_min"] == 3
    md = (tmp_path / "saidas" / "comparacao.md").read_text(encoding="utf-8")
    assert "| lambda0 | 1 | 0 | 0 | 0 | 1 | 0.45 |" in md and "—" in md
    assert (tmp_path / "saidas" / "comparacao.png").exists()


def test_mu_cresce_com_o_passo():
    from rsskd.train import mu_schedule

    assert mu_schedule(0, 100, 0.5) == 0.0
    assert mu_schedule(50, 100, 0.5) == pytest.approx(0.25)
    assert mu_schedule(100, 100, 0.5) == pytest.approx(0.5)
    # Com start_frac = 0,5: zero na primeira metade, depois sobe até 0,5.
    assert mu_schedule(40, 100, 0.5, start_frac=0.5) == 0.0
    assert mu_schedule(75, 100, 0.5, start_frac=0.5) == pytest.approx(0.25)


def test_passo_da_etapa_2_combina_as_perdas():
    # Aluno pequeno, sem pré-treino; o professor da etapa 1 é outro modelo
    # congelado. Com lr = 0 os pesos não mudam, e dá para conferir a soma.
    from rsskd.detector import build_student
    from rsskd.distill import freeze
    from rsskd.train import Stage2

    S = 64
    torch.manual_seed(0)
    professor = freeze(build_student(pretrained_backbone=False, image_size=S))
    torch.manual_seed(1)
    m = build_student(pretrained_backbone=False, image_size=S)
    masks = torch.zeros(1, S, S, dtype=torch.uint8)
    masks[0, 10:30, 10:30] = 1
    alvo = {"boxes": torch.tensor([[10.0, 10, 30, 30]]),
            "labels": torch.ones(1, dtype=torch.int64), "masks": masks}
    opt = make_optimizer(m, lr=0.0)
    imgs = [torch.randn(6, S, S), torch.randn(6, S, S)]

    cfg = Stage2(teacher=professor, kd_weight=2.0, mu=0.3, rule_area=150.0)
    out = train_step(m, opt, imgs, [alvo, alvo], "cpu", consistency="flip",
                     consistency_weight=3.0, stage2=cfg)
    for k in ["loss_kd", "kd_cls", "kd_mask", "loss_regra1", "regra1_satisfeita",
              "loss_consistencia", "mu"]:
        assert k in out
    outras = sum(out[k] for k in ["loss_box_reg", "loss_mask", "loss_objectness",
                                  "loss_rpn_box_reg"])
    esperado = ((1 - 0.3) * out["loss_classifier"] + outras + 0.3 * out["loss_regra1"]
                + 2.0 * out["loss_kd"] + 3.0 * out["loss_consistencia"])
    assert out["total"] == pytest.approx(esperado, rel=1e-4)
    assert out["loss_kd"] > 0 and out["loss_regra1"] > 0

    # mu = 0: a regra é medida, mas não pesa; sem professor, não há destilação.
    out = train_step(m, opt, imgs, [alvo, alvo], "cpu", stage2=Stage2(mu=0.0, rule_area=150.0))
    assert "loss_kd" not in out and out["mu"] == 0.0 and out["loss_regra1"] > 0
    outras = sum(out[k] for k in ["loss_classifier", "loss_box_reg", "loss_mask",
                                  "loss_objectness", "loss_rpn_box_reg"])
    assert out["total"] == pytest.approx(outras, rel=1e-4)
    # Sem área de referência, a regra não aparece.
    out = train_step(m, opt, imgs, [alvo, alvo], "cpu", stage2=Stage2(mu=0.5))
    assert "loss_regra1" not in out


def test_treino_da_etapa_2(dados_falsos, tmp_path):
    # Etapa 1 curtinha, depois etapa 2 com ela como professor e a regra 1.
    import csv

    from train_student import main

    dados, pseudo, entrada = dados_falsos
    base = ["--pasta", str(dados), "--copas", str(pseudo), "--entrada", str(entrada),
            "--lote", "1", "--sem-pretreino", "--dispositivo", "cpu",
            "--saida", str(tmp_path / "runs"), "--epocas", "2",
            "--max-passos-por-epoca", "1", "--aquecimento", "1", "--paciencia", "0"]
    etapa1 = main(["--nome", "e1"] + base)
    etapa2 = main(["--nome", "e2", "--professor", str(etapa1 / "melhor.pt"),
                   "--mu-max", "0.5", "--peso-kd", "2"] + base)
    cfg = json.loads((etapa2 / "config.json").read_text(encoding="utf-8"))
    assert cfg["etapa"] == 2 and cfg["area_regra_usada"] == 400   # caixa 20x20 do falso
    linhas = list(csv.DictReader(open(etapa2 / "historico.csv", encoding="utf-8")))
    assert {"loss_kd", "kd_cls", "kd_mask", "loss_regra1", "mu"} <= set(linhas[0])
    # mu sobe com o passo: 0 no 1o passo, metade no 2o (média de 1 passo/época).
    assert [float(r["mu"]) for r in linhas] == [0.0, 0.25]
