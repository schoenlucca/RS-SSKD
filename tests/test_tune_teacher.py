"""Ajuste visual do professor (scripts/tune_teacher.py)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("matplotlib")
from PIL import Image  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))
from rsskd.cerrado import BAND_ORDER, build_dataset  # noqa: E402
from tune_teacher import main, parse_grid  # noqa: E402


def _mata_fechada(pasta: Path, n: int = 2, lado: int = 512):
    """Fotos em que tudo é vegetação e as copas só se distinguem pelo brilho."""
    d = pasta / "voo"
    d.mkdir(parents=True)
    yy, xx = np.mgrid[:lado, :lado]
    nir = np.full((lado, lado), 90.0)
    for cy in range(40, lado, 85):
        for cx in range(40, lado, 85):
            nir += 120 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 22.0 ** 2))
    valores = {"R": np.full((lado, lado), 22.0), "G": nir * 0.4, "RE": nir * 0.7, "NIR": nir}
    for k in range(1, n + 1):
        for b in BAND_ORDER:
            Image.fromarray(np.clip(valores[b], 0, 255).astype(np.uint8)).save(
                d / f"processed-DJI_20250127100000_{k:04d}_MS_{b}.TIF")


def test_grade():
    combos = parse_grid(["split_by=distance,nir", "min_distance=10,20", "th=otsu"])
    assert len(combos) == 4
    assert combos[0] == {"split_by": "distance", "min_distance": 10, "th1": "otsu", "th2": "otsu"}
    with pytest.raises(ValueError):
        parse_grid(["min_distance"])


def test_figura_resumo_e_gravar(tmp_path):
    _mata_fechada(tmp_path / "proj")
    build_dataset(tmp_path / "proj", tmp_path / "cerrado", {"train": ["voo"]}, log=None)
    config = tmp_path / "prof.json"
    shutil.copy(RAIZ / "configs" / "cerrado_teacher.json", config)

    linhas = main(["--pasta", str(tmp_path / "cerrado"), "--config", str(config),
                   "--grade", "split_by=distance,nir", "min_distance=20", "sigma=4", "th=0.2",
                   "--n", "3", "--saida", str(tmp_path / "saidas")])
    assert (tmp_path / "saidas" / "ajuste_professor.png").exists()
    assert len(linhas) == 2 and all(0 <= r["cobertura_da_vegetacao"] <= 1 for r in linhas)
    # Na mata fechada, separar pelo brilho do NIR acha mais copas que pela distância.
    assert linhas[1]["copas_por_recorte"] > linhas[0]["copas_por_recorte"]

    main(["--config", str(config), "--grade", "split_by=distance,nir", "sigma=4",
          "--gravar", "1"])
    prof = json.loads(config.read_text(encoding="utf-8"))["professor"]
    assert prof["split_by"] == "nir" and prof["sigma"] == 4 and "_ajuste" in prof
    with pytest.raises(SystemExit):
        main(["--config", str(config), "--grade", "sigma=4", "--gravar", "5"])
    with pytest.raises(TypeError):                                   # parâmetro que não existe
        main(["--config", str(config), "--grade", "raio=4", "--gravar", "0"])
