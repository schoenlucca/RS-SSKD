"""Testes da previsão por blocos em imagens grandes (parte 7e)."""

import numpy as np
import pytest
from scipy import ndimage as ndi

from rsskd.tiling import crown_table, predict_large, tile_starts


def _disco(h, w, cy, cx, r):
    yy, xx = np.mgrid[:h, :w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2


def _por_componentes(bands):
    """'Previsor' de mentira: cada mancha conectada da banda 0 é uma copa."""
    return ndi.label(bands[0] > 0)[0]  # type: ignore[index]


def test_posicoes_dos_blocos():
    assert tile_starts(256, 256, 64) == [0]
    assert tile_starts(100, 256, 64) == [0]
    assert tile_starts(600, 256, 64) == [0, 192, 344]     # o último encosta no fim
    starts = tile_starts(1000, 256, 64)
    assert starts[0] == 0 and starts[-1] + 256 == 1000
    assert all(b - a <= 192 for a, b in zip(starts, starts[1:]))  # sobreposição >= 64
    with pytest.raises(ValueError):
        tile_starts(1000, 256, 256)


def test_copas_na_divisa_sao_contadas_uma_vez_e_inteiras():
    # Imagem de 600x600 com 25 copas espalhadas, várias em cima das divisas
    # entre blocos (192, 344, ...). Cada copa tem de sair uma vez, inteira.
    H = W = 600
    img = np.zeros((1, H, W), np.uint8)
    centros = [(y, x) for y in (40, 190, 300, 450, 560) for x in (40, 200, 330, 470, 560)]
    for cy, cx in centros:
        img[0][_disco(H, W, cy, cx, 22)] = 1
    labels = predict_large(img, _por_componentes, tile=256, overlap=64)
    assert labels.max() == len(centros)                    # sem repetir nem perder
    for cy, cx in centros:
        d = _disco(H, W, cy, cx, 22)
        ids = np.unique(labels[d])
        assert len(ids) == 1 and ids[0] > 0                # uma etiqueta só...
        assert (labels == ids[0]).sum() == d.sum()         # ...e a copa inteira
    # Igual a prever a imagem toda de uma vez.
    assert labels.max() == _por_componentes(img).max()


def test_imagem_menor_que_um_bloco():
    img = np.zeros((1, 100, 120), np.uint8)
    img[0][_disco(100, 120, 50, 60, 10)] = 1
    labels = predict_large(img, _por_componentes, tile=256)
    assert labels.shape == (100, 120) and labels.max() == 1


def test_tabela_das_copas():
    lab = np.zeros((20, 20), np.int32)
    lab[0:4, 0:5] = 1                                       # 20 pixels
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    tr = from_origin(500000.0, 8000000.0, 0.05, 0.05)       # pixel de 5 cm
    (row,) = crown_table(lab, 0.05, tr)
    assert row["area_px"] == 20 and row["area_m2"] == pytest.approx(20 * 0.0025)
    assert (row["linha"], row["coluna"]) == (1.5, 2.0)
    assert row["x"] == pytest.approx(500000 + 2.5 * 0.05)
    assert row["y"] == pytest.approx(8000000 - 2.0 * 0.05)
    assert rasterio is not None


def test_script_com_o_professor(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    from predict_orthomosaic import main

    # Ortomosaico sintético de 5 bandas: copas com NIR alto e vermelho baixo.
    H = W = 400
    img = np.full((5, H, W), 100, np.uint8)
    img[3] = 60                                            # NIR do solo
    for cy, cx in [(80, 80), (80, 300), (300, 200), (200, 330)]:
        d = _disco(H, W, cy, cx, 25)
        img[0][d], img[3][d] = 20, 200
    tif = tmp_path / "orto.tif"
    with rasterio.open(tif, "w", driver="GTiff", height=H, width=W, count=5, dtype="uint8",
                       crs="EPSG:31983", transform=from_origin(0, 0, 0.05, 0.05)) as dst:
        dst.write(img)
    cfg = tmp_path / "prof.json"
    cfg.write_text('{"professor": {"nir_band": 3, "red_band": 0, "th1": 0.2, "th2": 0.2,'
                   ' "min_distance": 10, "min_area": 100}}', encoding="utf-8")
    labels = main(["--professor", "--imagem", str(tif), "--config-professor", str(cfg),
                   "--saida", str(tmp_path / "saida")])
    assert labels.max() == 4
    with rasterio.open(tmp_path / "saida_copas.tif") as src:
        assert src.crs.to_epsg() == 31983 and src.read(1).max() == 4
    linhas = (tmp_path / "saida_copas.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(linhas) == 5 and "area_m2" in linhas[0]


def test_previsor_do_aluno():
    # O aluno de verdade (pequeno, sem pré-treino) numa imagem de 2x2 blocos.
    torch = pytest.importorskip("torch")
    from rsskd.detector import build_student
    from rsskd.inputs import compute_input_spec
    from rsskd.tiling import model_tile_predictor

    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (5, 100, 100)).astype(np.uint8)
    spec = compute_input_spec([img], 3, 0, ["b0", "b1", "b2", "b3", "b4", "ndvi"])
    torch.manual_seed(0)
    model = build_student(pretrained_backbone=False, image_size=64)
    labels = predict_large(img, model_tile_predictor(model, spec, score_thr=0.0),
                           tile=64, overlap=16)
    assert labels.shape == (100, 100) and labels.dtype == np.int32
