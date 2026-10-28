"""Resume as copas do professor gravadas e compara com a referência.

Uso:
    python scripts/summarize_pseudolabels.py

Para cada divisão, mostra lado a lado as copas do professor e as da
referência: quantas copas por imagem, a área típica e a fração da imagem
coberta. A comparação mostra os vieses do professor (por exemplo, copas
grandes demais quando ele junta árvores vizinhas).

Grava em <pasta das copas>/resumo.json as estatísticas e a mediana da área
das copas do professor no TREINO, que é o valor usado pela regra da área
(regra 1 de Harmon et al., 2022). A mediana vem só do treino e só das copas
do professor: a referência é reservada para a avaliação.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsskd.data import DEFAULT_DIR, AgmlTreeCrowns, mask_to_instances  # noqa: E402
from rsskd.pseudolabels import (  # noqa: E402
    PseudoLabelStore, summarize_instances, summarize_store,
)

DEFAULT_PSEUDO = Path("data") / "pseudo" / "agml"


def _line(name, s):
    c, a = s["copas_por_imagem"], s["area_px"]
    return (f"  {name:<11} {s['imagens']:>4} imagens  {s['copas']:>6} copas  "
            f"copas/imagem {c['media']:5.1f} (mín {c['minimo']}, máx {c['maximo']})  "
            f"área mediana {a['mediana']:6.0f} px (p10 {a['p10']:.0f}, p90 {a['p90']:.0f})  "
            f"cobertura {s['cobertura']:.0%}  sem copa {s['imagens_sem_copa']}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--copas", default=str(DEFAULT_PSEUDO),
                    help="pasta das copas do professor")
    ap.add_argument("--pasta", default=str(DEFAULT_DIR),
                    help="pasta do conjunto AgML (para a referência)")
    ap.add_argument("--divisoes", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args(argv)

    store = PseudoLabelStore(args.copas)
    print(f"Professor: {store.params()['professor']}\n")

    resumo = {"professor": store.params()["professor"], "divisoes": {}}
    for split in args.divisoes:
        prof = summarize_store(store, split)
        ds = AgmlTreeCrowns(args.pasta, split=split)
        ref = summarize_instances(mask_to_instances(ds[i].mask) for i in range(len(ds)))
        resumo["divisoes"][split] = {"professor": prof, "referencia": ref}
        print(f"{split}:")
        print(_line("professor", prof))
        print(_line("referência", ref))

    if "train" in resumo["divisoes"]:
        area = resumo["divisoes"]["train"]["professor"]["area_px"]["mediana"]
        resumo["area_mediana_treino_px"] = area
        print(f"\nMediana da área das copas do professor no treino: {area:.0f} px "
              f"(valor para a regra da área)")

    out = Path(args.copas) / "resumo.json"
    out.write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Resumo salvo em {out}")
    return resumo


if __name__ == "__main__":
    main()
