"""Tabelas do artigo em LaTeX (scripts/latex_tables.py)."""

from __future__ import annotations

import json

import pytest

from latex_tables import main, table

RESULTADO = {
    "split": "test", "tta": True,
    "resumos": {
        "TEBAKA": [{"f1": 0.516, "erro_contagem": 3.6, "area_kl": 0.44}],
        "Etapa_2": [{"f1": 0.52, "erro_contagem": 4.5, "area_kl": 0.09},
                    {"f1": 0.53, "erro_contagem": 4.9, "area_kl": 0.10}],
    },
    "diferenca_f1_tebaka": {"Etapa_2": {"diferenca": 0.009, "inf": -0.01, "sup": 0.028,
                                        "p": 0.4}},
}


def test_tabela_com_media_desvio_negrito_e_diferenca():
    tex = table(RESULTADO, ["f1", "erro_contagem", "area_kl"], "Teste", "tab:t")
    assert "\\begin{tabular}{lcccc}" in tex and "\\toprule" in tex
    assert "Etapa\\_2 & \\textbf{0.525 $\\pm$ 0.007}" in tex          # maior F1
    assert "\\textbf{3.6}" in tex                                      # menor erro: TEBAKA
    assert "+0.009 [-0.010; 0.028]" in tex and "0.009*" not in tex     # cruza o zero
    assert tex.count("\\\\") == 3                                      # cabeçalho + 2 linhas


def test_virgula_e_metrica_desconhecida():
    tex = table(RESULTADO, ["f1"], "T", "tab:t", decimal="virgula", digits=2)
    assert "0,52" in tex and "0.52" not in tex
    with pytest.raises(ValueError):
        table(RESULTADO, ["auc"], "T", "tab:t")


def test_script(tmp_path):
    p = tmp_path / "teste_final_tta.json"
    p.write_text(json.dumps(RESULTADO), encoding="utf-8")
    (out,) = main([str(p), "--metricas", "f1", "area_kl", "--saida", str(tmp_path / "tab")])
    tex = out.read_text(encoding="utf-8")
    assert out.name == "teste_final_tta.tex" and "\\label{tab:teste_final_tta}" in tex
    assert "com TTA" in tex
