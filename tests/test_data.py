"""Testes do carregador do conjunto AgML com um arquivo Parquet falso.

O arquivo falso imita o formato descrito na página do conjunto (5 bandas
uint8 de 256x256, máscara e RGB como imagens PNG, coluna split). Quando o
conjunto real for baixado, scripts/inspect_agml.py confere o formato real.
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
Image = pytest.importorskip("PIL.Image")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsskd.data import (  # noqa: E402
    AgmlTreeCrowns, decode_array, guess_nir_red, iou_binary, mask_to_instances,
    match_instances,
)

H = W = 64
NIR, RED = 4, 2   # ordem escolhida para o arquivo falso


def _png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return {"bytes": buf.getvalue(), "path": None}


def _fake_row(seed, split):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:H, :W]
    mask = (((yy - 20) ** 2 + (xx - 20) ** 2 <= 100)
            | ((yy - 45) ** 2 + (xx - 45) ** 2 <= 81)).astype(np.uint8)
    bands = rng.integers(60, 90, (5, H, W)).astype(np.uint8)  # "solo"
    bands[NIR][mask == 1] = 200    # copa: NIR alto
    bands[RED][mask == 1] = 20     # copa: vermelho baixo
    rgb = np.stack([bands[RED], bands[1], bands[0]], -1)
    return {"bands": bands.tobytes(), "bands_shape": [5, H, W],
            "bands_dtype": "uint8",
            "band_metadata": json.dumps({"crs": "EPSG:4326"}),
            "band_order": None, "mask": _png(mask * 255), "rgb": _png(rgb),
            "split": split}


@pytest.fixture()
def pasta(tmp_path):
    rows = [_fake_row(i, s) for i, s in enumerate(["train"] * 4 + ["val"] * 2)]
    pd.DataFrame(rows).to_parquet(tmp_path / "train-00000.parquet")
    return tmp_path


def test_carrega_e_filtra_por_divisao(pasta):
    assert len(AgmlTreeCrowns(pasta)) == 6
    ds = AgmlTreeCrowns(pasta, split="val")
    assert len(ds) == 2
    s = ds[0]
    assert s.bands.shape == (5, H, W) and s.bands.dtype == np.uint8
    assert s.mask.shape == (H, W) and set(np.unique(s.mask)) == {0, 1}
    assert s.rgb is not None and s.rgb.shape == (H, W, 3)
    assert s.metadata["crs"] == "EPSG:4326"


def test_decodifica_listas():
    arr = decode_array([[1, 2], [3, 4]], shape=[4], dtype="uint8")
    assert arr.tolist() == [1, 2, 3, 4]


def test_palpite_das_bandas(pasta):
    nir, red, _ = guess_nir_red(AgmlTreeCrowns(pasta))
    assert (nir, red) == (NIR, RED)


def test_instancias_e_metricas(pasta):
    ref = mask_to_instances(AgmlTreeCrowns(pasta)[0].mask)
    assert ref.max() == 2                                    # 2 copas separadas
    perfeito = match_instances(ref, ref)
    assert perfeito["tp"] == 2 and perfeito["f1"] == 1.0
    so_uma = np.where(ref == 1, 1, 0)                        # previu só 1 copa
    r = match_instances(so_uma, ref)
    assert r["tp"] == 1 and r["fn"] == 1 and r["fp"] == 0
    assert iou_binary(ref > 0, ref > 0) == 1.0


def test_mascara_com_um_numero_por_copa(tmp_path):
    # Rótulos à mão no Cerrado: copas encostadas, cada uma com seu número.
    # Como copa contra fundo, as duas virariam uma só; com os números, ficam 2.
    import io

    import pandas as pd
    from PIL import Image

    from rsskd.data import is_instance_mask

    m = np.zeros((H, W), np.uint16)
    m[4:20, 4:16] = 7
    m[4:20, 16:28] = 300                                     # encostada na copa 7
    assert is_instance_mask(m) and not is_instance_mask((m > 0).astype(np.uint8) * 255)
    assert mask_to_instances((m > 0).astype(np.uint8)).max() == 1
    ref = mask_to_instances(m)
    assert ref.max() == 2 and ref[10, 10] != ref[10, 20] and ref[0, 0] == 0

    buf = io.BytesIO()
    Image.fromarray(m).save(buf, format="PNG")              # PNG de 16 bits
    bandas = np.zeros((4, H, W), np.uint8)
    pd.DataFrame([{"bands": bandas.tobytes(), "bands_shape": list(bandas.shape),
                   "bands_dtype": "uint8", "mask": buf.getvalue(), "split": "test"}]
                 ).to_parquet(tmp_path / "test.parquet")
    s = AgmlTreeCrowns(tmp_path, split="test")[0]
    assert s.mask.max() == 300                               # os números chegam inteiros
    assert mask_to_instances(s.mask).max() == 2
