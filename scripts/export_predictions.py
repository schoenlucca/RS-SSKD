"""Exporta as previsões de um aluno treinado para uma pasta (parte 7c).

Uso:
    python scripts/export_predictions.py --run runs/lambda100 --divisoes val test
    python scripts/export_predictions.py --run runs/etapa2 --saida previsoes/etapa2
    python scripts/export_predictions.py --run runs/etapa2 --tta   # em previsoes/etapa2_tta

Grava, no formato de rsskd/predictions.py, uma imagem de etiquetas por
recorte (0 = fundo, 1..N = copas) e um info.json com o modelo usado e a
lista das imagens (com a soma de verificação das bandas). É o mesmo formato
que o Felipe entrega, e o que scripts/evaluate_predictions.py avalia.

Com --tta, cada recorte é previsto nas 8 orientações e as copas são
combinadas (rsskd/tta.py); a pasta padrão ganha o sufixo _tta.

Também serve para entregar as nossas previsões a quem quiser medi-las com
outro código.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsskd.dataset import CrownDataset  # noqa: E402
from rsskd.evaluation import predict_crowns  # noqa: E402
from rsskd.inputs import InputSpec  # noqa: E402
from rsskd.predictions import PredictionStore, split_manifest  # noqa: E402
from rsskd.train import load_checkpoint, pick_device  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="pasta do treino (ex.: runs/lambda100)")
    ap.add_argument("--modelo", default="melhor",
                    help="melhor, ultimo, melhor_referencia ou epocas/epoca_NN")
    ap.add_argument("--divisoes", nargs="+", default=["val", "test"])
    ap.add_argument("--saida", default=None, help="padrão: previsoes/<nome do treino>")
    ap.add_argument("--limiar-score", type=float, default=0.5)
    ap.add_argument("--dispositivo", default="auto")
    ap.add_argument("--tta", action="store_true",
                    help="combina as previsões das 8 orientações (rotações e espelho)")
    ap.add_argument("--tta-votos", type=int, default=None,
                    help="mínimo de orientações em que a copa aparece (padrão: 4)")
    args = ap.parse_args(argv)

    run_dir = Path(args.run)
    device = pick_device(args.dispositivo)
    model, cfg, ckpt = load_checkpoint(run_dir / f"{args.modelo}.pt", device)
    spec = InputSpec.load(cfg["entrada"])
    model.eval()
    default = Path("previsoes") / (run_dir.name + ("_tta" if args.tta else ""))
    store = PredictionStore(args.saida or default)

    manifests = {}
    with torch.no_grad():
        for split in args.divisoes:
            ds = CrownDataset(split, cfg["copas"], spec, dataset_dir=cfg["pasta"])
            for i in range(len(ds)):
                image, _ = ds[i]
                store.save(split, i, predict_crowns(model, image, device, args.limiar_score,
                                                    tta=args.tta, tta_votes=args.tta_votos))
            manifests[split] = split_manifest(ds.images)
            print(f"{split}: {len(ds)} imagens exportadas")
    tta = " com TTA" if args.tta else ""
    store.write_info(metodo=f"aluno {run_dir.name}{tta} ({args.modelo}.pt, época {ckpt['epoca']})",
                     limiar_score=args.limiar_score, tta=args.tta, tta_votos=args.tta_votos,
                     imagens=manifests)
    print(f"Previsões em {store.root}/")
    return store.root


if __name__ == "__main__":
    main()
