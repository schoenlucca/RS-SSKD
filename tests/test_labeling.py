"""Recortes para rotular (rsskd/labeling.py, scripts/export_for_labeling.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rsskd.cerrado import BAND_ORDER
from rsskd.labeling import choose_crops, export_crops, false_color, load_manifest


def _voo(pasta: Path, nome: str, n: int, lado: int = 600) -> Path:
    d = pasta / nome
    d.mkdir(parents=True)
    for k in range(1, n + 1):
        for i, b in enumerate(BAND_ORDER):
            a = np.full((lado, lado), 30 * (i + 1), np.uint8)
            a[:, : lado // 2] += 50                          # metade esquerda mais clara
            Image.fromarray(a).save(d / f"processed-DJI_20250127100000_{k:04d}_MS_{b}.TIF")
    return d


def test_escolha_espalhada_e_repetivel(tmp_path):
    _voo(tmp_path, "A", 40)
    crops = choose_crops(tmp_path, ["A"], n=4, size=512, gap=10, seed=3)
    nums = sorted(int(c.disparo.split("_")[1]) for c in crops)
    assert len(nums) == 4 and all(b - a >= 10 for a, b in zip(nums, nums[1:]))
    assert all(c.y0 % 2 == 0 and c.x0 % 2 == 0 and c.y0 + 512 <= 600 for c in crops)
    assert crops == choose_crops(tmp_path, ["A"], n=4, size=512, gap=10, seed=3)
    with pytest.raises(ValueError, match="Só cabem"):
        choose_crops(tmp_path, ["A"], n=10, gap=10)
    with pytest.raises(ValueError, match="múltiplo"):
        choose_crops(tmp_path, ["A"], n=1, size=511)


def test_exportar_imagens_e_manifesto(tmp_path):
    _voo(tmp_path / "proj", "A", 12)
    crops = choose_crops(tmp_path / "proj", ["A"], n=2, gap=5)
    manif = export_crops(tmp_path / "proj", crops, tmp_path / "rot")
    for c in crops:
        im = np.asarray(Image.open(tmp_path / "rot" / "imagens" / c.arquivo))
        assert im.shape == (512, 512, 3) and im.dtype == np.uint8
    origem, lidos = load_manifest(manif)
    assert origem == str(tmp_path / "proj") and lidos == crops


def test_falsa_cor_coloca_o_nir_no_vermelho():
    b = np.zeros((4, 8, 8), np.uint8)
    b[BAND_ORDER.index("NIR"), :, :4] = 200
    fc = false_color(b)
    assert fc[0, 0, 0] == 255 and fc[0, 7, 0] == 0 and fc[..., 1].max() == 0


def test_script_recusa_voo_do_treino(tmp_path):
    from export_for_labeling import main

    _voo(tmp_path / "proj", "A", 3)
    _voo(tmp_path / "proj", "B", 3)
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "manifesto.json").write_text(json.dumps(
        {"divisoes": {"train": {"voos": {"A": {}}}, "val": {"voos": {}},
                      "test": {"voos": {}}}}), encoding="utf-8")
    comum = ["--origem", str(tmp_path / "proj"), "--pasta", str(tmp_path / "c"),
             "--n", "1", "--saida", str(tmp_path / "rot")]
    with pytest.raises(SystemExit):
        main(["--voos", "A"] + comum)
    assert len(main(["--voos", "B"] + comum)) == 1
