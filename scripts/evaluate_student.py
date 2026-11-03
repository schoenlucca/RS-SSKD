"""Avalia um aluno já treinado, na validação ou no teste.

Uso:
    python scripts/evaluate_student.py --run runs/lambda1
    python scripts/evaluate_student.py --run runs/lambda1 --modelo ultimo
    python scripts/evaluate_student.py --run runs/lambda1 --split test

Carrega o modelo salvo pelo scripts/train_student.py (melhor.pt ou
ultimo.pt), roda nas imagens da divisão escolhida e mede, com as mesmas
métricas do professor:
    - contra a referência (o resultado que vai para o relatório);
    - contra as copas do professor TEBAKA (o quanto o aluno o imita).

**O teste é para o fim.** Use a validação para comparar versões e escolher
parâmetros; rode no teste só as versões finais, uma vez, para o relatório.
Escolher qualquer coisa olhando o teste deixaria o número otimista.

Saídas em runs/<nome>/:
    avaliacao_<divisão>_<modelo>.json   resumo das métricas
    avaliacao_<divisão>_<modelo>.csv    uma linha por imagem
    avaliacao_<divisão>_<modelo>.png    exemplos: imagem, professor, aluno, referência
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsskd.dataset import CrownDataset  # noqa: E402
from rsskd.detector import build_student  # noqa: E402
from rsskd.evaluation import (  # noqa: E402
    CrownMetrics, evaluate_student, format_summary, predictions_to_instances,
)
from rsskd.inputs import InputSpec  # noqa: E402
from rsskd.train import pick_device  # noqa: E402
from overfit_check import colorize  # noqa: E402
from show_batch import rgb_from_input  # noqa: E402


def load_student(run_dir: Path, which: str, device):
    """Recria o aluno com a configuração do treino e carrega os pesos."""
    ckpt = torch.load(run_dir / f"{which}.pt", map_location=device, weights_only=False)
    cfg = ckpt["config"]
    spec = InputSpec.load(cfg["entrada"])
    model = build_student(spec.in_channels, not cfg["sem_pretreino"])
    model.load_state_dict(ckpt["modelo"])
    return model.to(device).eval(), spec, cfg, ckpt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="pasta do treino (ex.: runs/lambda1)")
    ap.add_argument("--modelo", default="melhor", choices=["melhor", "ultimo"])
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--limiar-score", type=float, default=0.5,
                    help="confiança mínima para uma detecção virar copa")
    ap.add_argument("--dispositivo", default="auto")
    ap.add_argument("--n-figura", type=int, default=4)
    args = ap.parse_args(argv)

    run_dir = Path(args.run)
    device = pick_device(args.dispositivo)
    model, spec, cfg, ckpt = load_student(run_dir, args.modelo, device)
    ds = CrownDataset(args.split, cfg["copas"], spec, dataset_dir=cfg["pasta"])
    print(f"{run_dir.name}/{args.modelo}.pt (época {ckpt['epoca']}), "
          f"{len(ds)} imagens de {args.split}")

    vs_teacher = CrownMetrics()
    s_ref, m_ref = evaluate_student(model, ds, device, score_thr=args.limiar_score,
                                    teacher_metrics=vs_teacher)
    s_prof = vs_teacher.summary()
    print(f"Aluno x referência: {format_summary(s_ref)}")
    print(f"Aluno x professor:  {format_summary(s_prof)}")

    tag = f"avaliacao_{args.split}_{args.modelo}"
    (run_dir / f"{tag}.json").write_text(json.dumps(
        {"modelo": args.modelo, "epoca": ckpt["epoca"], "split": args.split,
         "limiar_score": args.limiar_score, "aluno_x_referencia": s_ref,
         "aluno_x_professor": s_prof}, indent=2, ensure_ascii=False), encoding="utf-8")
    import csv
    with open(run_dir / f"{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(m_ref.rows[0]))
        w.writeheader()
        w.writerows(m_ref.rows)

    # Figura com alguns exemplos.
    idx = np.linspace(0, len(ds) - 1, min(args.n_figura, len(ds))).astype(int)
    fig, ax = plt.subplots(len(idx), 4, figsize=(12, 3 * len(idx)), squeeze=False)
    with torch.no_grad():
        for r, i in enumerate(idx.tolist()):
            image, _ = ds[i]
            (out,) = model([image.to(device)])
            pred = predictions_to_instances(out, args.limiar_score)
            teacher, ref = ds.store.load(args.split, i), ds.reference(i)
            panels = [(rgb_from_input(image, spec), f"imagem {i}"),
                      (colorize(teacher), f"professor: {teacher.max()} copas"),
                      (colorize(pred), f"aluno: {pred.max()} copas"),
                      (colorize(ref), f"referência: {ref.max()} copas")]
            for a, (img, title) in zip(ax[r], panels):
                a.imshow(img)
                a.set_title(title, fontsize=9)
                a.axis("off")
    fig.tight_layout()
    fig.savefig(run_dir / f"{tag}.png", dpi=100)
    plt.close(fig)
    print(f"Resultados em {run_dir}/{tag}.*")
    return s_ref, s_prof


if __name__ == "__main__":
    main()
