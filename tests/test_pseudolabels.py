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
