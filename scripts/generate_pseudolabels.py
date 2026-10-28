"""Gera as copas do professor TEBAKA (pseudo-rótulos) e grava no disco.

Uso:
    python scripts/generate_pseudolabels.py
    python scripts/generate_pseudolabels.py --divisoes train --processos 4

Por padrão, roda o professor em todas as imagens das três divisões (train,
val e test) do conjunto AgML, com os parâmetros de configs/agml_teacher.json,
e grava em data/pseudo/agml/ (ignorada pelo git).

As copas do treino são as que o aluno vai usar como alvo. As de validação e
teste servem para comparar o professor com o aluno nas mesmas imagens; o
aluno nunca é avaliado contra elas, e sim contra as máscaras de referência.

Se a geração for interrompida, rodar de novo continua de onde parou: as
imagens que já têm copas gravadas são puladas.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsskd.data import DEFAULT_DIR, AgmlTreeCrowns  # noqa: E402
from rsskd.pseudolabels import PseudoLabelStore  # noqa: E402
from rsskd.teacher import TebakaTeacher  # noqa: E402

DEFAULT_OUT = Path("data") / "pseudo" / "agml"


def _run_teacher(args):
    """Roda o professor numa imagem. Fica fora do main para poder ser
    executada em outro processo (com --processos maior que 1)."""
    teacher, index, bands = args
    return index, teacher(bands)["instances"]


def generate(dataset_dir, out_dir, config, splits, processes=1, log=print):
    """Gera e grava as copas. Devolve quantas imagens foram geradas por divisão."""
    teacher = TebakaTeacher.from_config(config)
    store = PseudoLabelStore(out_dir, teacher)   # confere os parâmetros
    generated = {}

    for split in splits:
        ds = AgmlTreeCrowns(dataset_dir, split=split)
        done = set(store.indices(split))
        todo = [i for i in range(len(ds)) if i not in done]
        log(f"{split}: {len(ds)} imagens, {len(done)} já geradas, "
            f"{len(todo)} a gerar")
        if not todo:
            generated[split] = 0
            continue

        start = time.time()
        jobs = ((teacher, i, ds[i].bands) for i in todo)
        if processes > 1:
            # Vários processos rodam o professor em imagens diferentes ao
            # mesmo tempo. Cada imagem é independente, então é seguro.
            with ProcessPoolExecutor(max_workers=processes) as pool:
                results = pool.map(_run_teacher, jobs, chunksize=4)
                for n, (i, inst) in enumerate(results, 1):
                    store.save(split, i, inst)
                    if n % 50 == 0 or n == len(todo):
                        log(f"  {n}/{len(todo)} ({time.time() - start:.0f} s)")
        else:
            for n, job in enumerate(jobs, 1):
                i, inst = _run_teacher(job)
                store.save(split, i, inst)
                if n % 50 == 0 or n == len(todo):
                    log(f"  {n}/{len(todo)} ({time.time() - start:.0f} s)")
        generated[split] = len(todo)
    return generated


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default=str(DEFAULT_DIR),
                    help="pasta do conjunto AgML baixado")
    ap.add_argument("--saida", default=str(DEFAULT_OUT),
                    help="pasta onde as copas são gravadas")
    ap.add_argument("--config", default="configs/agml_teacher.json")
    ap.add_argument("--divisoes", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--processos", type=int, default=1,
                    help="quantos processos em paralelo (ex.: 4)")
    args = ap.parse_args(argv)

    generated = generate(args.pasta, args.saida, args.config, args.divisoes,
                         args.processos)
    print(f"Pronto. Geradas agora: {generated}. Copas em {args.saida}")
    return generated


if __name__ == "__main__":
    main()
