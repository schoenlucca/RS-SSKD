"""Testes da figura qualitativa (parte 7d)."""

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("matplotlib")

from rsskd.data import AgmlTreeCrowns, mask_to_instances  # noqa: E402
from rsskd.predictions import PredictionStore  # noqa: E402
from test_data import _fake_row  # noqa: E402


def test_contornos_sobre_a_imagem():
    from figure_predictions import overlay, parse_methods

    rgb = np.zeros((20, 20, 3), np.float32)
    lab = np.zeros((20, 20), np.int32)
    lab[5:15, 5:15] = 1
    out = overlay(rgb, lab)
    assert out[5, 10].tolist() == pytest.approx([1.0, 0.85, 0.0])   # borda pintada
    assert out[10, 10].tolist() == [0, 0, 0]                          # miolo intacto
    assert out[2, 2].tolist() == [0, 0, 0]                            # fundo intacto
    assert parse_methods(["Etapa 1=runs/x", "previsoes/y"]) == [
        ("Etapa 1", "runs/x"), ("y", "previsoes/y")]


def test_figura_com_dois_metodos(tmp_path):
    from figure_predictions import main as desenhar

    dados = tmp_path / "dados"
    dados.mkdir()
    pd.DataFrame([_fake_row(i, "test") for i in range(3)]).to_parquet(dados / "t.parquet")
    ds = AgmlTreeCrowns(dados, split="test")
    a, b = PredictionStore(tmp_path / "a"), PredictionStore(tmp_path / "b")
    for i in range(3):
        ref = mask_to_instances(ds[i].mask)
        a.save("test", i, ref)
        b.save("test", i, np.zeros_like(ref))
    out = desenhar([f"Perfeito={tmp_path / 'a'}", str(tmp_path / "b"), "--split", "test",
                    "--pasta", str(dados), "--n", "2", "--saida", str(tmp_path / "s")])
    assert all(p.exists() and p.stat().st_size > 5_000 for p in out)
    assert [p.suffix for p in out] == [".png", ".pdf"]
