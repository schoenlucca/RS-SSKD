"""Testes do professor TEBAKA com uma imagem sintética. Rodar com: pytest

A imagem imita um recorte de drone com 4 bandas (verde, vermelho,
Red-Edge, NIR): solo com reflectância parecida no vermelho e no NIR, e
copas redondas com NIR alto e vermelho baixo (NDVI alto).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rsskd.teacher import (  # noqa: E402
    TebakaTeacher, compute_ndvi, instances_to_targets, median_crown_area,
)

H = W = 256
NIR, RED = 3, 1


def _disk(cy, cx, r):
    yy, xx = np.mgrid[:H, :W]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2


def synthetic_image(seed=0):
    """6 copas: 4 isoladas e 2 encostadas (para testar o watershed)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 4), np.float32)
    img[..., RED] = 0.20 + rng.normal(0, 0.02, (H, W))   # solo
    img[..., NIR] = 0.25 + rng.normal(0, 0.02, (H, W))
    crowns = [(50, 50, 25), (50, 190, 22), (190, 50, 20), (200, 200, 24),
              (120, 105, 22), (120, 147, 22)]                 # as 2 últimas se tocam
    for cy, cx, r in crowns:
        d = _disk(cy, cx, r)
        img[d, RED] = 0.05 + rng.normal(0, 0.01, d.sum())
        img[d, NIR] = 0.50 + rng.normal(0, 0.03, d.sum())
    # Pontos soltos de "vegetação" que devem ser removidos como ruído.
    for _ in range(30):
        y, x = rng.integers(0, H, 2)
        img[y, x, RED], img[y, x, NIR] = 0.05, 0.5
    return img, len(crowns)


def test_ndvi_valores():
    ndvi = compute_ndvi(np.array([0.5, 0.25]), np.array([0.05, 0.20]))
    assert np.allclose(ndvi, [0.818, 0.111], atol=1e-3)


def test_professor_encontra_todas_as_copas():
    img, n = synthetic_image()
    out = TebakaTeacher(nir_band=NIR, red_band=RED)(img)
    assert out["n_crowns"] == n
    # O ruído (pixels isolados) não pode virar copa.
    assert out["vegetation"].sum() < 0.30 * H * W


def test_copas_encostadas_sao_separadas():
    img, _ = synthetic_image()
    inst = TebakaTeacher(nir_band=NIR, red_band=RED)(img)["instances"]
    # Os centros das duas copas encostadas têm de cair em etiquetas diferentes.
    a, b = inst[120, 105], inst[120, 147]
    assert a > 0 and b > 0 and a != b


def test_canais_primeiro():
    img, n = synthetic_image()
    out = TebakaTeacher(nir_band=NIR, red_band=RED)(np.moveaxis(img, -1, 0))
    assert out["n_crowns"] == n


def test_alvos_mask_rcnn_e_mediana():
    img, n = synthetic_image()
    inst = TebakaTeacher(nir_band=NIR, red_band=RED)(img)["instances"]
    t = instances_to_targets(inst)
    assert t["masks"].shape == (n, H, W)
    assert t["boxes"].shape == (n, 4)
    assert (t["boxes"][:, 2] > t["boxes"][:, 0]).all()
    # Copas com raio 20 a 25: área entre ~1250 e ~2000 pixels.
    assert 1000 < median_crown_area([inst]) < 2200


def test_limiar_otsu():
    # Somar 0,3 ao vermelho derruba o NDVI de tudo (as copas caem para perto
    # de 0,2), como acontece com bandas sem calibração. O limiar fixo de 0,4
    # não acharia nada; o limiar automático (Otsu) se adapta e acha as copas.
    img, n = synthetic_image()
    img[..., RED] = img[..., RED] + 0.3
    out = TebakaTeacher(nir_band=NIR, red_band=RED, th1="otsu", th2="otsu")(img)
    assert out["n_crowns"] == n


def test_configuracao_do_professor(tmp_path):
    import json

    import pytest

    # O arquivo do repositório cria o professor com os parâmetros calibrados.
    cfg = Path(__file__).resolve().parents[1] / "configs" / "agml_teacher.json"
    t = TebakaTeacher.from_config(cfg)
    assert (t.nir_band, t.red_band, t.min_distance) == (3, 0, 18)

    # Um parâmetro com erro de digitação tem de gerar erro, e não ser ignorado.
    ruim = tmp_path / "ruim.json"
    ruim.write_text(json.dumps({"professor": {"nir_bnad": 3}}), encoding="utf-8")
    with pytest.raises(KeyError):
        TebakaTeacher.from_config(ruim)
