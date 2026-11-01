"""Avalia o professor TEBAKA contra as máscaras de referência do conjunto AgML.

Uso:
    # Avaliar com os parâmetros calibrados (padrão) no conjunto de teste
    python scripts/evaluate_teacher.py --split test

    # Busca em grade dos parâmetros no conjunto de validação
    python scripts/evaluate_teacher.py --split val --grade

O que ele mede (critério de Harmon et al., 2022):
  - por copa: precisão, revocação e F1, com acerto quando a copa prevista
    tem IoU >= 0,5 com uma copa da referência; e o IoU médio dos acertos;
  - por pixel: IoU da máscara de vegetação contra a máscara de referência;
  - contagem: erro absoluto médio entre o número de copas previsto e o real.

Saídas em saidas/: uma tabela CSV com o resultado de cada imagem e uma
figura com exemplos (imagem, referência e copas do professor).

Atenção: em parte das imagens a referência marca só algumas árvores (as
camélias do estudo original). Árvores reais não anotadas contam como
falsos positivos, então a precisão fica subestimada. A revocação (das copas
anotadas, quantas o professor encontrou) não sofre com isso.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsskd.data import DEFAULT_DIR, AgmlTreeCrowns, mask_to_instances  # noqa: E402
from rsskd.evaluation import CrownMetrics, format_summary  # noqa: E402
from rsskd.teacher import TebakaTeacher  # noqa: E402

# Os parâmetros calibrados do professor ficam neste arquivo, lido também
# pelos outros scripts. Para mudar um parâmetro, mude só lá.
DEFAULT_CONFIG = Path("configs/agml_teacher.json")


def evaluate(samples, refs, teacher):
    """Roda o professor em todas as amostras e soma as métricas.

    As métricas vêm de rsskd/evaluation.py, as mesmas usadas no aluno.
    """
    metrics, outputs = CrownMetrics(), []
    for s, ref in zip(samples, refs):
        out = teacher(s.bands)
        metrics.add(out["instances"], ref, index=s.index,
                    pred_mask=out["vegetation"])
        outputs.append(out)
    return metrics.summary(), metrics.rows, outputs


fmt = format_summary


def figure(samples, refs, outputs, path, n=4):
    """Figura com n exemplos: imagem, referência e copas do professor."""
    idx = np.linspace(0, len(samples) - 1, n).astype(int)
    fig, ax = plt.subplots(n, 3, figsize=(9, 3 * n))
    rng = np.random.default_rng(0)
    for row, i in enumerate(idx):
        s, ref, out = samples[i], refs[i], outputs[i]
        inst = out["instances"]
        colors = rng.random((max(inst.max(), ref.max()) + 1, 3))
        colors[0] = 0
        ax[row, 0].imshow(s.rgb if s.rgb is not None else s.bands[:3].transpose(1, 2, 0))
        ax[row, 0].set_title(f"imagem {s.index}")
        ax[row, 1].imshow(colors[ref])
        ax[row, 1].set_title(f"referência: {ref.max()} copas")
        ax[row, 2].imshow(colors[inst])
        ax[row, 2].set_title(f"professor: {inst.max()} copas")
        for a in ax[row]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--pasta", default=str(DEFAULT_DIR))
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--grade", action="store_true",
                    help="busca em grade dos parâmetros (use com --split val)")
    args = ap.parse_args()

    ds = AgmlTreeCrowns(args.pasta, split=args.split)
    samples = [ds[i] for i in range(len(ds))]
    refs = [mask_to_instances(s.mask) for s in samples]
    print(f"{len(samples)} imagens ({args.split}), "
          f"{sum(r.max() for r in refs)} copas na referência")

    base = TebakaTeacher.from_config(args.config)

    if args.grade:
        # Parte dos parâmetros do arquivo e varia só os três principais.
        results = []
        for th, md, ma in itertools.product([-0.5, -0.45, -0.42, -0.38, -0.3, "otsu"],
                                            [12, 18, 25], [150, 300]):
            teacher = replace(base, th1=th, th2=th, min_distance=md, min_area=ma)
            summ, _, _ = evaluate(samples, refs, teacher)
            results.append((summ["f1"], th, md, ma))
            print(f"th={th} min_distance={md} min_area={ma}: {fmt(summ)}", flush=True)
        f1, th, md, ma = max(results, key=lambda r: r[0])
        print(f"\nMelhor F1 = {f1:.2f} com th={th}, min_distance={md}, min_area={ma}")
        return

    summary, rows, outputs = evaluate(samples, refs, base)
    print(f"Parâmetros ({args.config}): th1={base.th1}, th2={base.th2}, "
          f"min_distance={base.min_distance}, min_area={base.min_area}, "
          f"NIR=banda {base.nir_band}, vermelho=banda {base.red_band}")
    print(fmt(summary))

    out_dir = Path("saidas")
    out_dir.mkdir(exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_dir / f"professor_{args.split}.csv", index=False)
    figure(samples, refs, outputs, out_dir / f"professor_{args.split}.png")
    print(f"Tabela e figura salvas em {out_dir}/professor_{args.split}.*")


if __name__ == "__main__":
    main()
