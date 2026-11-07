"""Copas de um ortomosaico inteiro, com o aluno ou com o professor (parte 7e).

Uso:
    # Um GeoTIFF com as 5 bandas na ordem do AgML (vermelho, verde, azul, NIR, Red-Edge)
    python scripts/predict_orthomosaic.py --run runs/etapa2 --imagem orto.tif

    # Um arquivo por banda (o Mavic 3M grava assim), na mesma ordem
    python scripts/predict_orthomosaic.py --run runs/etapa2 \
        --bandas R.tif G.tif B.tif NIR.tif RE.tif

    # O professor TEBAKA, em vez do aluno
    python scripts/predict_orthomosaic.py --professor --imagem orto.tif

Recorta a imagem em blocos de 256x256 sobrepostos, prevê cada bloco e junta
as copas sem repetir as da divisa (rsskd/tiling.py). Saídas, ao lado de
--saida (padrão: saidas/orto_<nome>):
    <saida>_copas.tif   imagem de etiquetas (0 = fundo, 1..N = copas), com a
                        mesma georreferência da entrada: abre no QGIS por
                        cima do ortomosaico
    <saida>_copas.csv   uma linha por copa: área (px e m²) e centro
                        (linha/coluna e coordenadas no mapa)

**Atenção às bandas.** O aluno foi treinado com as 5 bandas do AgML, nessa
ordem e em 8 bits, normalizadas pelas estatísticas do AgML. Imagens de outra
câmera ou em reflectância precisam de cuidado: a ordem tem de ser a mesma, e
valores em outra escala mudam o que a rede vê. Para as imagens do drone do
projeto, o caminho certo é gerar as copas do professor, recalcular a
normalização e treinar nelas (as mesmas etapas, com outro conjunto).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsskd.inputs import InputSpec  # noqa: E402
from rsskd.paths import require  # noqa: E402
from rsskd.teacher import TebakaTeacher  # noqa: E402
from rsskd.tiling import (  # noqa: E402
    TilePredictor, crown_table, model_tile_predictor, predict_large,
)


def read_bands(image: str | None, band_files: list[str] | None) -> tuple[np.ndarray, dict]:
    """Bandas (C, H, W) e o perfil do GeoTIFF (para gravar a saída alinhada)."""
    import rasterio

    hint = ("Use o caminho do seu arquivo: os nomes dos exemplos (orto.tif, R.tif, ...) "
            "são só ilustrações.")
    for f in [image] if image else (band_files or []):
        require(f, "a imagem", hint)
    if image:
        with rasterio.open(image) as src:
            return src.read(), dict(src.profile)
    assert band_files
    arrays, profile = [], None
    for path in band_files:
        with rasterio.open(path) as src:
            arrays.append(src.read(1))
            profile = profile or dict(src.profile)
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise ValueError(f"As bandas têm tamanhos diferentes: {shapes}. Alinhe-as antes "
                         f"(mesma grade de pixels).")
    assert profile is not None
    return np.stack(arrays), profile


def main(argv=None):
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--imagem", help="GeoTIFF com todas as bandas")
    src.add_argument("--bandas", nargs="+", help="um GeoTIFF por banda, na ordem do AgML")
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--run", help="pasta do treino do aluno (usa melhor.pt)")
    who.add_argument("--professor", action="store_true", help="usa o professor TEBAKA")
    ap.add_argument("--config-professor", default="configs/agml_teacher.json")
    ap.add_argument("--bloco", type=int, default=256)
    ap.add_argument("--sobreposicao", type=int, default=64)
    ap.add_argument("--limiar-score", type=float, default=0.5)
    ap.add_argument("--dispositivo", default="auto")
    ap.add_argument("--saida", default=None)
    args = ap.parse_args(argv)

    bands, profile = read_bands(args.imagem, args.bandas)
    print(f"Imagem: {bands.shape[0]} bandas, {bands.shape[1]} x {bands.shape[2]} pixels")

    if args.professor:
        teacher = TebakaTeacher.from_config(args.config_professor)

        def teacher_tile(b: np.ndarray) -> np.ndarray:
            return teacher(b)["instances"]

        predict: TilePredictor = teacher_tile
        name = "professor"
    else:
        from rsskd.train import load_checkpoint, pick_device

        device = pick_device(args.dispositivo)
        model, cfg, _ = load_checkpoint(Path(args.run) / "melhor.pt", device)
        spec = InputSpec.load(cfg["entrada"])
        n_bands = spec.in_channels - int(spec.add_ndvi)
        if bands.shape[0] != n_bands:
            raise ValueError(f"O aluno espera {n_bands} bandas; a imagem tem {bands.shape[0]}.")
        predict = model_tile_predictor(model, spec, str(device), args.limiar_score)
        name = Path(args.run).name

    labels = predict_large(bands, predict, args.bloco, args.sobreposicao, log=print)
    out = Path(args.saida or f"saidas/orto_{name}")
    out.parent.mkdir(parents=True, exist_ok=True)

    import rasterio

    meta = dict(profile)
    meta.update(count=1, dtype="uint32", nodata=None, compress="deflate")
    with rasterio.open(f"{out}_copas.tif", "w", **meta) as dst:
        dst.write(labels.astype(np.uint32), 1)
    transform = profile.get("transform")
    pixel = abs(transform.a) if transform is not None else None
    rows = crown_table(labels, pixel, transform)
    with open(f"{out}_copas.csv", "w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"{len(rows)} copas. Saídas: {out}_copas.tif e {out}_copas.csv")
    return labels


if __name__ == "__main__":
    main()
