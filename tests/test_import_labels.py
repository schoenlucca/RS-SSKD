"""Rótulos à mão em COCO (rsskd/coco.py, scripts/import_labels.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")
from test_labeling import _voo  # noqa: E402

from rsskd.coco import downscale_labels, original_name, rasterize  # noqa: E402


def _quadrado(x0, y0, lado):
    return [[x0, y0, x0 + lado, y0, x0 + lado, y0 + lado, x0, y0 + lado]]


def test_nome_original_do_roboflow():
    assert original_name("A_2025_0007_10_20_png.rf.3f9a2b.jpg") == "A_2025_0007_10_20.png"
    assert original_name("pasta/A_1.png") == "A_1.png"


def test_poligonos_viram_copas_e_a_menor_fica_por_cima():
    anns = [{"segmentation": _quadrado(10, 10, 20)},             # pequena, dentro da grande
            {"segmentation": _quadrado(0, 0, 60)}]
    lab = rasterize(anns, 64, 64)
    assert lab.max() == 2 and lab[20, 20] != lab[5, 5] and lab[5, 5] > 0
    with pytest.raises(ValueError, match="RLE"):
        rasterize([{"segmentation": {"counts": "x", "size": [64, 64]}}], 64, 64)


def test_reducao_mantem_copas_encostadas_separadas():
    lab = np.zeros((8, 8), np.int32)
    lab[:, :4], lab[:, 4:] = 3, 9                               # duas copas lado a lado
    lab[0, 0] = 5                                               # copa de 1 pixel
    small = downscale_labels(lab, 2, min_area=1)
    assert small.shape == (4, 4) and small.max() == 2
    assert len(np.unique(small[:, :2])) == 1 and small[0, 0] != small[0, 3]


def test_importar_como_teste(tmp_path):
    from export_for_labeling import main as exportar
    from import_labels import main as importar

    from rsskd.data import AgmlTreeCrowns, mask_to_instances

    _voo(tmp_path / "proj", "T", 12)
    crops = exportar(["--voos", "T", "--origem", str(tmp_path / "proj"), "--n", "2",
                      "--espaco", "5", "--pasta", str(tmp_path / "c"),
                      "--saida", str(tmp_path / "rot")])
    # Imagem 1: duas copas encostadas. Imagem 2: exportada pelo Roboflow,
    # renomeada e reduzida para 256 (os polígonos vêm na metade do tamanho).
    a, b = crops
    coco = {"images": [{"id": 1, "file_name": a.arquivo, "width": 512, "height": 512},
                       {"id": 2, "width": 256, "height": 256,
                        "file_name": b.arquivo[:-4] + "_png.rf.ab12.jpg"}],
            "annotations": [{"image_id": 1, "segmentation": _quadrado(100, 100, 100)},
                            {"image_id": 1, "segmentation": _quadrado(200, 100, 100)},
                            {"image_id": 2, "segmentation": _quadrado(20, 20, 50)}]}
    (tmp_path / "rot" / "rotulos.json").write_text(json.dumps(coco), encoding="utf-8")
    args = ["--coco", str(tmp_path / "rot" / "rotulos.json"),
            "--manifesto", str(tmp_path / "rot" / "manifesto.json"),
            "--pasta", str(tmp_path / "c"), "--rotulador", "teste"]
    rows = importar(args)
    assert len(rows) == 2

    ds = AgmlTreeCrowns(tmp_path / "c", split="test")
    s0, s1 = ds[0], ds[1]
    assert s0.bands.shape == (4, 256, 256)
    ref0 = mask_to_instances(s0.mask)
    assert ref0.max() == 2                                      # encostadas, mas separadas
    assert (ref0 == 1).sum() == pytest.approx(50 * 50, rel=0.05)
    assert mask_to_instances(s1.mask).max() == 1
    assert (s1.mask > 0).sum() == pytest.approx(50 * 50, rel=0.1)  # 256 -> 512 -> 256
    assert s0.metadata["rotulador"] == "teste" and s0.metadata["voo"] == "T"
    manif = json.loads((tmp_path / "c" / "manifesto.json").read_text(encoding="utf-8"))
    assert manif["divisoes"]["test"]["voos"]["T"] == {"recortes": 2, "copas": 3}

    with pytest.raises(SystemExit, match="já existe"):
        importar(args)
    assert len(importar(args + ["--substituir"])) == 2

