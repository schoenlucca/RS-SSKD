"""Exporta os recortes do Cerrado para rotular à mão.

Uso, no servidor do LISA:
    python3 scripts/export_for_labeling.py --voos <voo reservado para o teste> --n 50
    python3 scripts/export_for_labeling.py --voos 2025-06-16 --n 10 --saida rotulagem/piloto

Grava em rotulagem/ (ou --saida):
    imagens/<voo>_<disparo>_<y0>_<x0>.png   recortes de 512×512 em falsa cor
    manifesto.json                           de onde veio cada recorte

As imagens vão para a ferramenta de rotulação (CVAT ou Roboflow), uma
copa = um polígono. O manifesto fica guardado: é ele que liga cada imagem de
volta à foto e à posição de origem na hora de importar os rótulos
(scripts/import_labels.py). Detalhes da escolha em rsskd/labeling.py; o
protocolo de rotulação está em docs/protocolo_rotulacao.md.

O script recusa voos que estão no treino ou na validação de --pasta (pelo
manifesto.json de scripts/prepare_cerrado.py): a referência é o teste, e
rotular imagens do treino deixaria o resultado otimista.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsskd.labeling import choose_crops, export_crops  # noqa: E402
from rsskd.paths import require  # noqa: E402

ORIGEM = "/mnt/sdb-seagate/graduacao/datasets/projeto_cerrado"


def used_flights(dataset_dir: Path) -> dict[str, str]:
    """{voo: divisão} dos voos de treino e validação do conjunto preparado."""
    path = dataset_dir / "manifesto.json"
    if not path.exists():
        return {}
    m = json.loads(path.read_text(encoding="utf-8"))
    return {voo: split for split, s in m["divisoes"].items() if split in ("train", "val")
            for voo in s["voos"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--voos", nargs="+", required=True, help="voos de onde saem os recortes")
    ap.add_argument("--origem", default=ORIGEM, help="pasta com uma subpasta por voo")
    ap.add_argument("--n", type=int, default=50, help="quantos recortes")
    ap.add_argument("--tamanho", type=int, default=512,
                    help="lado do recorte em pixels da foto original")
    ap.add_argument("--escala", type=float, default=0.5,
                    help="redução usada no modelo (a de prepare_cerrado.py)")
    ap.add_argument("--espaco", type=int, default=10,
                    help="distância mínima entre os números dos disparos escolhidos")
    ap.add_argument("--semente", type=int, default=0)
    ap.add_argument("--pasta", default="data/cerrado",
                    help="conjunto preparado, para recusar voos do treino e da validação")
    ap.add_argument("--saida", default="rotulagem")
    args = ap.parse_args(argv)

    require(args.origem, "a pasta das fotos do Cerrado", "Confira --origem.")
    clash = {v: s for v, s in used_flights(Path(args.pasta)).items() if v in args.voos}
    if clash:
        raise SystemExit(f"Voos usados no treino ou na validação: {clash}. "
                         "A referência precisa vir de voos que o modelo nunca viu.")
    crops = choose_crops(args.origem, args.voos, args.n, args.tamanho, args.escala,
                         args.espaco, args.semente)
    manifest = export_crops(args.origem, crops, args.saida)
    print(f"{len(crops)} recortes de {args.tamanho}×{args.tamanho} em {args.saida}/imagens/")
    print(f"Manifesto: {manifest} (guarde junto com os rótulos)")
    return crops


if __name__ == "__main__":
    main()
