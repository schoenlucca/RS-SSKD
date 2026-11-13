"""TTA nos scripts de avaliação e exportação."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")
from test_train import dados_falsos  # noqa: E402,F401  (fixture)


def test_avaliar_exportar_e_teste_final_com_tta(dados_falsos, tmp_path):
    from evaluate_student import main as avaliar
    from export_predictions import main as exportar
    from final_test import main as final
    from train_student import main as treinar

    from rsskd.predictions import PredictionStore

    dados, pseudo, entrada = dados_falsos
    run = treinar(["--nome", "t", "--pasta", str(dados), "--copas", str(pseudo),
                   "--entrada", str(entrada), "--lote", "1", "--sem-pretreino",
                   "--dispositivo", "cpu", "--saida", str(tmp_path / "runs"), "--epocas", "1",
                   "--max-passos-por-epoca", "1", "--paciencia", "0",
                   "--peso-consistencia", "0"])

    avaliar(["--run", str(run), "--dispositivo", "cpu", "--tta", "--n-figura", "1"])
    resumo = json.loads((run / "avaliacao_val_melhor_tta.json").read_text(encoding="utf-8"))
    assert resumo["tta"] is True

    pasta = exportar(["--run", str(run), "--divisoes", "val", "--dispositivo", "cpu",
                      "--tta", "--saida", str(tmp_path / "prev_tta")])
    store = PredictionStore(pasta)
    assert store.info()["tta"] is True and store.load("val", 0).shape == (64, 64)

    final([f"Aluno={tmp_path / 'runs' / 't'}", "--split", "val", "--pasta", str(dados),
           "--copas", str(pseudo), "--previsoes", str(tmp_path / "prev"),
           "--saida", str(tmp_path / "saidas"), "--dispositivo", "cpu", "--bootstrap", "100",
           "--tta"])
    assert (tmp_path / "prev" / "t_tta" / "info.json").exists()
    assert (tmp_path / "saidas" / "final_val_tta.md").exists()
