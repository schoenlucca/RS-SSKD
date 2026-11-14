"""Importa os rótulos à mão (COCO) como a divisão de teste do Cerrado.

Uso, depois de rotular as imagens de scripts/export_for_labeling.py:
    python3 scripts/import_labels.py --coco rotulagem/rotulos.json --rotulador "Lucca"
    python3 scripts/generate_pseudolabels.py --pasta data/cerrado --config configs/cerrado_teacher.json \\
        --saida data/pseudo/cerrado --divisoes test
    python3 scripts/final_test.py "Etapa 1=runs/cerrado_l0_s*" --pasta data/cerrado \\
        --copas data/pseudo/cerrado

Para cada imagem rotulada:
  1. acha no manifesto (rotulagem/manifesto.json) de qual foto e posição ela
     veio;
  2. transforma os polígonos em copas numeradas (rsskd/coco.py) e reduz para
     a escala do modelo (512 -> 256);
  3. lê as 4 bandas da mesma região, na escala do modelo, e grava o recorte
     no formato do AgML, com a referência como máscara de 16 bits (um número
     por copa: copas encostadas continuam separadas na avaliação).

O resultado vai para <pasta>/test.parquet (padrão data/cerrado/test.parquet),
e o manifesto.json do conjunto passa a listar os voos do teste. Daí em
diante, o teste do Cerrado é avaliado como o do AgML: professor, aluno,
SAM do Felipe, todos contra a mesma referência.

Imagens do manifesto sem rótulo no COCO são avisadas e ficam de fora. Voos
que estão no treino ou na validação são recusados.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_for_labeling import used_flights  # noqa: E402

from rsskd.cerrado import BAND_NAMES, BAND_ORDER, find_captures, read_capture  # noqa: E402
from rsskd.coco import downscale_labels, load_coco  # noqa: E402
from rsskd.labeling import MANIFEST, load_manifest  # noqa: E402
from rsskd.paths import require  # noqa: E402


def png16(labels: np.ndarray) -> bytes:
    from PIL import Image

    if labels.max() > 65535:
        raise ValueError("Mais de 65535 copas num recorte.")
    buf = io.BytesIO()
    Image.fromarray(labels.astype(np.uint16)).save(buf, format="PNG")
    return buf.getvalue()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, help="arquivo COCO exportado (polígonos)")
    ap.add_argument("--manifesto", default=f"rotulagem/{MANIFEST}")
    ap.add_argument("--origem", default=None,
                    help="pasta das fotos (padrão: a gravada no manifesto)")
    ap.add_argument("--pasta", default="data/cerrado", help="conjunto preparado")
    ap.add_argument("--divisao", default="test", choices=["test", "val"])
    ap.add_argument("--rotulador", default="", help="quem rotulou (vai para os metadados)")
    ap.add_argument("--min-area", type=int, default=10,
                    help="copas menores que isso (em pixels do modelo) são descartadas")
    ap.add_argument("--substituir", action="store_true",
                    help="sobrescreve uma divisão que já existe")
    args = ap.parse_args(argv)

    import pandas as pd

    source, crops = load_manifest(require(args.manifesto, "o manifesto dos recortes",
                                          "Ele é gravado por scripts/export_for_labeling.py."))
    source = args.origem or source
    pasta = Path(args.pasta)
    out = pasta / f"{args.divisao}.parquet"
    if out.exists() and not args.substituir:
        raise SystemExit(f"{out} já existe; use --substituir para trocar.")
    clash = {v for c in crops for v, s in used_flights(pasta).items() if v == c.voo}
    if clash:
        raise SystemExit(f"Recortes de voos do treino ou da validação: {sorted(clash)}.")

    size = crops[0].tamanho
    labels = load_coco(require(args.coco, "o arquivo COCO"), size)
    rows, missing, per_flight = [], [], {}
    captures: dict[str, dict] = {}
    for c in crops:
        if c.arquivo not in labels:
            missing.append(c.arquivo)
            continue
        factor = round(1 / c.escala)
        ref = downscale_labels(labels[c.arquivo], factor, args.min_area)
        if c.voo not in captures:
            captures[c.voo] = {k.key: k for k in find_captures(Path(source) / c.voo)}
        bands = read_capture(captures[c.voo][c.disparo], c.escala)
        y0, x0, t = c.y0 // factor, c.x0 // factor, c.tamanho // factor
        piece = np.ascontiguousarray(bands[:, y0:y0 + t, x0:x0 + t])
        meta = {"voo": c.voo, "disparo": c.disparo, "y0": y0, "x0": x0, "escala": c.escala,
                "recorte": t, "arquivo": c.arquivo, "rotulador": args.rotulador,
                "rotulos": Path(args.coco).name}
        rows.append({"bands": piece.tobytes(), "bands_shape": list(piece.shape),
                     "bands_dtype": "uint8", "band_metadata": json.dumps(meta),
                     "band_order": json.dumps([BAND_NAMES[b] for b in BAND_ORDER]),
                     "mask": png16(ref), "rgb": None, "split": args.divisao})
        per_flight.setdefault(c.voo, {"recortes": 0, "copas": 0})
        per_flight[c.voo]["recortes"] += 1
        per_flight[c.voo]["copas"] += int(ref.max())
    if missing:
        print(f"Aviso: {len(missing)} imagens do manifesto sem rótulo no COCO ficaram de fora "
              f"(ex.: {missing[0]}).")
    if not rows:
        raise SystemExit("Nenhuma imagem do manifesto tem rótulo no COCO. Confira os nomes.")

    pasta.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    manifest_path = pasta / "manifesto.json"
    m = (json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists()
         else {"divisoes": {}})
    m["divisoes"][args.divisao] = {"voos": per_flight, "recortes": len(rows),
                                   "referencia": {"coco": str(args.coco),
                                                  "rotulador": args.rotulador}}
    manifest_path.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    n_crowns = sum(f["copas"] for f in per_flight.values())
    print(f"{len(rows)} recortes rotulados ({n_crowns} copas, "
          f"{n_crowns / len(rows):.1f} por recorte) em {out}")
    return rows


if __name__ == "__main__":
    main()
