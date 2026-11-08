"""Bootstrap das métricas (rsskd/stats.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsskd.stats import (  # noqa: E402
    bootstrap_ci, counts_from_rows, format_ci, metric_from_counts, paired_bootstrap,
)


def _rows(tps, fps, fns):
    return [{"indice": i, "tp": t, "fp": p, "fn": n, "n_pred": t + p, "n_ref": t + n}
            for i, (t, p, n) in enumerate(zip(tps, fps, fns))]


def test_metrica_igual_ao_resumo():
    rows = _rows([8, 5, 0], [2, 5, 1], [2, 0, 3])
    c = counts_from_rows(rows)
    tp, fp, fn = 13, 8, 5
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    assert metric_from_counts(c, "precisao") == pytest.approx(prec)
    assert metric_from_counts(c, "revocacao") == pytest.approx(rec)
    assert metric_from_counts(c, "f1") == pytest.approx(2 * prec * rec / (prec + rec))
    assert metric_from_counts(c, "erro_contagem") == pytest.approx((0 + 5 + 2) / 3)
    with pytest.raises(ValueError):
        metric_from_counts(c, "auc")


def test_intervalo_contem_o_valor_e_encolhe_com_mais_imagens():
    rng = np.random.default_rng(1)
    def amostra(n):
        tp = rng.integers(3, 10, n)
        return _rows(tp, rng.integers(0, 5, n), rng.integers(0, 5, n))
    pouco, muito = bootstrap_ci(amostra(20), n_boot=2000), bootstrap_ci(amostra(400), n_boot=2000)
    for ci in (pouco, muito):
        assert ci["inf"] <= ci["valor"] <= ci["sup"]
    assert (muito["sup"] - muito["inf"]) < (pouco["sup"] - pouco["inf"])
    # Mesma semente, mesmo resultado.
    fixas = amostra(20)
    assert bootstrap_ci(fixas, seed=3) == bootstrap_ci(fixas, seed=3)


def test_pareado_ve_diferenca_consistente_e_ignora_ruido_comum():
    rng = np.random.default_rng(0)
    n = 60
    tp = rng.integers(2, 20, n)                 # imagens fáceis e difíceis, comuns aos dois
    fp, fn = rng.integers(0, 4, n), rng.integers(0, 4, n)
    a = _rows(tp + 1, fp, fn)                   # A acerta uma copa a mais em toda imagem
    b = _rows(tp, fp, fn + 1)
    r = paired_bootstrap(a, b, n_boot=4000)
    assert r["diferenca"] > 0 and r["inf"] > 0 and r["p"] < 0.01
    igual = paired_bootstrap(a, a, n_boot=1000)
    assert igual["diferenca"] == 0 and igual["p"] == 1.0


def test_pareado_exige_as_mesmas_imagens():
    a = _rows([1, 2, 3], [0, 0, 0], [0, 0, 0])
    with pytest.raises(ValueError):
        paired_bootstrap(a, a[:2])
    b = [dict(r, indice=r["indice"] + 10) for r in a]
    with pytest.raises(ValueError):
        paired_bootstrap(a, b)


def test_formato():
    assert format_ci({"valor": 0.5214, "inf": 0.49, "sup": 0.551}) == "0,521 [0,490; 0,551]"
    assert format_ci({"diferenca": 0.014, "inf": -0.002, "sup": 0.03}).startswith("+0,014")
