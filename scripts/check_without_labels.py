"""Diagnóstico do aluno sem referência (Cerrado, antes dos rótulos à mão).

Uso:
    python3 scripts/check_without_labels.py "runs/cerrado_l0_s*"
    python3 scripts/check_without_labels.py "runs/cerrado_l0_s*" --n 300 --split val

Sem referência não há F1 de verdade, mas dá para medir se o aluno é
**consistente**, o que é uma condição necessária (não suficiente) para ele
ser bom. Para cada treino (cada semente), nas mesmas --n imagens da divisão:

  1. Concordância com o professor: F1 por copa do aluno contra as copas do
     TEBAKA (a mesma conta do treino). Perto de 1 = o aluno copia o
     professor; bem abaixo = ele se afasta (para melhor ou para pior: só a
     referência dirá).
  2. Estabilidade sob rotação e espelhamento: o aluno prevê a imagem girada
     (90°, 180°) e espelhada; a previsão é desfeita de volta à orientação
     original e comparada (F1 por copa) com a previsão da imagem original.
     Uma copa de verdade não depende de como a foto foi girada: um aluno que
     aprendeu copas dá quase o mesmo resultado; um que aprendeu ruído, não.
  3. Contagem e tamanho: copas por imagem e área mediana, ao lado das do
     professor.

E, entre as sementes, a concordância de cada par (F1 por copa de uma contra
a outra): sementes que concordam entre si aprenderam a mesma coisa, e não o
acaso da inicialização.

Para comparar, a estabilidade é medida também no professor (o TEBAKA
aplicado na imagem girada e desfeito de volta).

Com --tta, o aluno prevê cada imagem nas 8 orientações combinadas
(rsskd/tta.py). A estabilidade deixa de ser medida: o TTA usa as mesmas 8
orientações em qualquer imagem girada, então dá o mesmo resultado por
construção. O que interessa com TTA é se a concordância entre as sementes e
com o professor sobe. As saídas ganham o sufixo _tta.

Saídas em saidas/: sem_referencia_<divisão>.md e .json.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsskd.augment import dihedral, dihedral_torch, inverse_dihedral  # noqa: E402
from rsskd.dataset import CrownDataset  # noqa: E402
from rsskd.evaluation import CrownMetrics, predict_crowns, predictions_to_instances  # noqa: E402
from rsskd.inputs import InputSpec  # noqa: E402
from rsskd.paths import MissingFileError  # noqa: E402
from rsskd.teacher import TebakaTeacher  # noqa: E402
from rsskd.train import load_checkpoint, pick_device  # noqa: E402

# Rotação de 90°, de 180° e espelhamento (k, espelha).
TRANSFORMS = [(1, False), (2, False), (0, True)]


def f1_between(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    """F1 por copa de a contra b, nas mesmas imagens (é simétrico)."""
    m = CrownMetrics()
    for i, (x, y) in enumerate(zip(a, b)):
        m.add(x, y, index=i)
    return float(m.summary()["f1"])


def stats(labels: list[np.ndarray]) -> dict:
    areas = [a for lab in labels for a in np.bincount(lab.ravel())[1:] if a > 0]
    return {"copas_por_imagem": float(np.mean([lab.max() for lab in labels])),
            "area_mediana_px": float(np.median(areas)) if areas else 0.0}


@torch.no_grad()
def student_predictions(model, images, device, transform=None, score_thr=0.5,
                        tta=False, tta_votes=None):
    """Copas do aluno em cada imagem; com transform=(k, espelha), prevê na
    imagem transformada e desfaz a transformação nas copas. Com tta, cada
    imagem é prevista nas 8 orientações combinadas."""
    if tta:
        return [predict_crowns(model, x, device, score_thr, tta=True, tta_votes=tta_votes)
                for x in images]
    out = []
    for x in images:
        if transform:
            x = dihedral_torch(x, *transform)
        (o,) = model([x.to(device)])
        lab = predictions_to_instances(o, score_thr)
        if transform:
            lab = inverse_dihedral(torch.from_numpy(lab), *transform).numpy()
        out.append(np.ascontiguousarray(lab))
    return out


def teacher_rotated(teacher: TebakaTeacher, bands: list[np.ndarray], transform) -> list:
    """O professor na imagem transformada, desfeito de volta."""
    out = []
    for b in bands:
        lab = teacher(np.ascontiguousarray(dihedral(b, *transform)))["instances"]
        out.append(inverse_dihedral(torch.from_numpy(lab), *transform).numpy())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("treinos", nargs="+", help='pastas ou padrões, ex.: "runs/cerrado_l0_s*"')
    ap.add_argument("--modelo", default="melhor")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=200, help="imagens usadas (espalhadas na divisão)")
    ap.add_argument("--config-professor", default=None,
                    help="configuração do TEBAKA, para medir a estabilidade dele "
                         "(padrão: o parametros.json das copas usadas no treino)")
    ap.add_argument("--limiar-score", type=float, default=0.5)
    ap.add_argument("--dispositivo", default="auto")
    ap.add_argument("--saida", default="saidas")
    ap.add_argument("--tta", action="store_true",
                    help="o aluno prevê com as 8 orientações combinadas (rsskd/tta.py)")
    ap.add_argument("--tta-votos", type=int, default=None)
    args = ap.parse_args(argv)

    runs = sorted({Path(p) for pat in args.treinos for p in glob.glob(pat) if Path(p).is_dir()})
    if not runs:
        raise MissingFileError(f"Nenhum treino casa com {args.treinos}.")
    device = pick_device(args.dispositivo)

    # Os treinos do grupo usam o mesmo conjunto: as imagens vêm do 1º.
    _, cfg, _ = load_checkpoint(runs[0] / f"{args.modelo}.pt", "cpu")
    spec = InputSpec.load(cfg["entrada"])
    ds = CrownDataset(args.split, cfg["copas"], spec, dataset_dir=cfg["pasta"])
    idx = np.unique(np.linspace(0, len(ds) - 1, min(args.n, len(ds))).astype(int)).tolist()
    images = [ds[i][0] for i in idx]
    teacher_labels = [ds.store.load(args.split, i) for i in idx]
    print(f"{len(runs)} treinos, {len(idx)} imagens de {args.split} ({cfg['pasta']})")

    rows, by_run = [], {}
    for run in runs:
        model, _, ckpt = load_checkpoint(run / f"{args.modelo}.pt", device)
        model.eval()
        base = student_predictions(model, images, device, None, args.limiar_score,
                                   args.tta, args.tta_votos)
        by_run[run.name] = base
        row = {"metodo": run.name, "epoca": ckpt["epoca"],
               "f1_x_professor": f1_between(base, teacher_labels)} | stats(base)
        if not args.tta:                                  # com TTA, é 1 por construção
            row["estabilidade"] = statistics.mean(
                f1_between(student_predictions(model, images, device, t, args.limiar_score), base)
                for t in TRANSFORMS)
        rows.append(row)
        print(f"{run.name}: x professor {row['f1_x_professor']:.3f}, "
              f"estabilidade {row.get('estabilidade', float('nan')):.3f}, "
              f"{row['copas_por_imagem']:.1f} copas/imagem")

    # O professor, para comparar.
    prof_cfg = Path(args.config_professor or ds.store.params_path)
    prof_row = {"metodo": "TEBAKA", "epoca": "", "f1_x_professor": 1.0} | stats(teacher_labels)
    if prof_cfg.exists():
        teacher = TebakaTeacher.from_config(prof_cfg)
        bands = [ds.images[i].bands for i in idx]
        prof_row["estabilidade"] = statistics.mean(
            f1_between(teacher_rotated(teacher, bands, t), teacher_labels) for t in TRANSFORMS)
    rows.insert(0, prof_row)

    pairs = {f"{a} x {b}": f1_between(by_run[a], by_run[b])
             for a, b in itertools.combinations(by_run, 2)}

    def cell(v, spec):
        return format(v, spec) if isinstance(v, (int, float)) and v != "" else "—"

    tta = ", aluno com TTA" if args.tta else ""
    lines = [f"Diagnóstico sem referência em {args.split} ({len(idx)} imagens de "
             f"{cfg['pasta']}{tta}).",
             "", "| Método | Época | F1 x professor | Estabilidade (rotação/espelho) | "
             "Copas por imagem | Área mediana (px) |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['metodo']} | {r['epoca']} | {cell(r['f1_x_professor'], '.3f')} | "
                     f"{cell(r.get('estabilidade', ''), '.3f')} | "
                     f"{cell(r['copas_por_imagem'], '.1f')} | {cell(r['area_mediana_px'], '.0f')} |")
    if pairs:
        lines += ["", "Concordância entre sementes (F1 por copa de uma contra a outra):", "",
                  "| Par | F1 |", "|---|---|"] + [f"| {k} | {v:.3f} |" for k, v in pairs.items()]
    out = Path(args.saida)
    out.mkdir(parents=True, exist_ok=True)
    md = "\n".join(lines) + "\n"
    tag = f"sem_referencia_{args.split}" + ("_tta" if args.tta else "")
    (out / f"{tag}.md").write_text(md, encoding="utf-8")
    (out / f"{tag}.json").write_text(json.dumps(
        {"imagens": idx, "metodos": rows, "pares_de_sementes": pairs},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(md)
    return rows, pairs


if __name__ == "__main__":
    main()
