"""Testes do armazenamento das copas do professor (pseudo-rótulos)."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsskd.pseudolabels import PseudoLabelStore  # noqa: E402
from rsskd.teacher import TebakaTeacher  # noqa: E402


def _copas():
    inst = np.zeros((64, 64), np.int32)
    inst[5:20, 5:20] = 1
    inst[30:50, 30:55] = 2
    return inst


def test_grava_e_le(tmp_path):
    store = PseudoLabelStore(tmp_path, TebakaTeacher())
    store.save("train", 3, _copas())
    store.save("train", 0, _copas())
    assert store.exists("train", 3) and not store.exists("val", 3)
    assert np.array_equal(store.load("train", 3), _copas())
    assert store.load("train", 3).dtype == np.int32
    assert store.indices("train") == [0, 3]
    assert store.indices("test") == []


def test_parametros_gravados(tmp_path):
    PseudoLabelStore(tmp_path, TebakaTeacher(th1=-0.42, min_distance=18))
    info = json.loads((tmp_path / "parametros.json").read_text(encoding="utf-8"))
    assert info["professor"]["th1"] == -0.42
    assert info["professor"]["min_distance"] == 18
    assert "criado_em" in info


def test_recusa_professor_diferente(tmp_path):
    t = TebakaTeacher(th1=-0.42)
    PseudoLabelStore(tmp_path, t)                      # cria a pasta
    PseudoLabelStore(tmp_path, t)                      # mesmo professor: ok
    with pytest.raises(ValueError, match="th1"):
        PseudoLabelStore(tmp_path, replace(t, th1=-0.30))


def test_script_de_geracao(tmp_path):
    # Usa o arquivo Parquet falso dos testes do carregador.
    pytest.importorskip("pyarrow")
    pd = pytest.importorskip("pandas")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from generate_pseudolabels import generate
    from test_data import _fake_row

    dados = tmp_path / "dados"
    dados.mkdir()
    rows = [_fake_row(i, s) for i, s in enumerate(["train"] * 3 + ["val"] * 2)]
    pd.DataFrame(rows).to_parquet(dados / "train-00000.parquet")

    # Professor com os índices de banda do arquivo falso (NIR 4, vermelho 2).
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"professor": {
        "nir_band": 4, "red_band": 2, "th1": 0.2, "th2": 0.2,
        "min_distance": 5, "min_area": 30}}), encoding="utf-8")

    saida = tmp_path / "pseudo"
    msgs = []
    assert generate(dados, saida, cfg, ["train", "val"], log=msgs.append) == {
        "train": 3, "val": 2}
    store = PseudoLabelStore(saida)
    assert store.indices("train") == [0, 1, 2]
    assert store.load("val", 1).max() == 2          # as 2 copas do arquivo falso

    # Rodar de novo não refaz nada: as imagens já geradas são puladas.
    assert generate(dados, saida, cfg, ["train", "val"], log=msgs.append) == {
        "train": 0, "val": 0}
