"""Leitura das fotos do Cerrado (rsskd/cerrado.py, scripts/prepare_cerrado.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarrow")
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsskd.cerrado import (  # noqa: E402
    BAND_ORDER, build_dataset, find_captures, read_capture, tiles,
)
from rsskd.data import AgmlTreeCrowns  # noqa: E402

# Valor de cada banda na foto sintética: dá para conferir a ordem depois.
VALOR = {"R": 40, "G": 80, "RE": 120, "NIR": 200}


def _voo(pasta: Path, nome: str, n: int, lado: int = 600, faltando: str | None = None):
    d = pasta / nome
    d.mkdir(parents=True)
    for k in range(1, n + 1):
        for b in BAND_ORDER:
            if faltando and k == n and b == faltando:
                continue
            a = np.full((lado, lado), VALOR[b], np.uint8)
            a[:10, :10] = k                                  # marca o disparo
            Image.fromarray(a).save(d / f"processed-DJI_20250127100000_{k:04d}_MS_{b}.TIF")
        Image.fromarray(np.zeros((lado, lado, 3), np.uint8)).save(
            d / f"processed-DJI_20250127100000_{k:04d}_D.JPG")     # ignorado
    return d


def test_disparos_e_ordem_das_bandas(tmp_path):
    d = _voo(tmp_path, "voo1", 3, faltando="RE")
    caps = find_captures(d)
    assert [c.number for c in caps] == ["0001", "0002"]       # o 3º sem RE fica de fora
    img = read_capture(caps[0], scale=0.5)
    assert img.shape == (4, 300, 300) and img.dtype == np.uint8
    assert [int(img[i, 100, 100]) for i in range(4)] == [VALOR[b] for b in BAND_ORDER]


def test_recortes_cobrem_a_imagem():
    img = np.zeros((4, 500, 500), np.uint8)
    t = list(tiles(img, 256, 12))
    assert len(t) == 4 and {(y, x) for y, x, _ in t} == {(0, 0), (0, 244), (244, 0), (244, 244)}
    assert all(r.shape == (4, 256, 256) for _, _, r in t)


def test_conjunto_no_formato_do_agml(tmp_path):
    origem = tmp_path / "projeto"
    _voo(origem, "A", 3)
    _voo(origem, "B", 2)
    _voo(origem, "C", 1)
    resumo = build_dataset(origem, tmp_path / "cerrado",
                           {"train": ["A", "B"], "val": ["C"], "test": []},
                           scale=1.0, tile=256, overlap=12, log=None)
    # 600 px: 3 posições por eixo -> 9 recortes por disparo.
    assert resumo["divisoes"]["train"]["recortes"] == 9 * 5
    assert resumo["divisoes"]["val"]["voos"] == {"C": {"disparos": 1, "recortes": 9}}
    assert not (tmp_path / "cerrado" / "test.parquet").exists()
    manif = json.loads((tmp_path / "cerrado" / "manifesto.json").read_text(encoding="utf-8"))
    assert manif["bandas"] == ["vermelho", "verde", "red_edge", "nir"]

    # O leitor do AgML lê sem mudança nenhuma.
    ds = AgmlTreeCrowns(tmp_path / "cerrado", split="train")
    assert len(ds) == 45
    s = ds[0]
    assert s.bands.shape == (4, 256, 256) and s.mask.shape == (256, 256)
    assert s.mask.max() == 0                                   # sem referência
    assert s.metadata["voo"] == "A" and s.metadata["disparo"].endswith("_0001")
    assert int(s.bands[3, 100, 100]) == VALOR["NIR"]
    assert len(AgmlTreeCrowns(tmp_path / "cerrado", split="val")) == 9


def test_um_voo_so_pode_estar_numa_divisao(tmp_path):
    _voo(tmp_path / "p", "A", 1)
    with pytest.raises(ValueError, match="mais de uma divisão"):
        build_dataset(tmp_path / "p", tmp_path / "o", {"train": ["A"], "val": ["A"]}, log=None)


def test_a_cada_e_professor_com_a_configuracao(tmp_path):
    from rsskd.teacher import TebakaTeacher

    origem = tmp_path / "projeto"
    _voo(origem, "A", 4)
    resumo = build_dataset(origem, tmp_path / "c", {"train": ["A"]}, scale=0.5,
                           every=2, log=None)
    assert resumo["divisoes"]["train"]["voos"]["A"]["disparos"] == 2
    # A configuração do Cerrado carrega e aponta para as bandas certas.
    prof = TebakaTeacher.from_config(Path(__file__).resolve().parents[1]
                                     / "configs" / "cerrado_teacher.json")
    assert prof.nir_band == 3 and prof.red_band == 0


def test_script(tmp_path):
    from prepare_cerrado import main

    origem = tmp_path / "projeto"
    _voo(origem, "A", 1, lado=512)
    _voo(origem, "B", 1, lado=512)
    resumo = main(["--origem", str(origem), "--saida", str(tmp_path / "c"),
                   "--treino", "A", "--val", "B"])
    assert resumo["divisoes"]["train"]["recortes"] == 1      # 512 -> 256: um recorte
    with pytest.raises(FileNotFoundError):
        main(["--origem", str(tmp_path / "nada")])
