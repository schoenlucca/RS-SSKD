"""Previsões em arquivo, para comparar métodos (parte 7c).

Para comparar o nosso aluno com o do Felipe (SAM), os dois precisam ser
medidos pelo mesmo código, nas mesmas imagens e contra a mesma referência.
Em vez de combinar métricas, combinamos um **formato de previsões**, o mesmo
das copas do professor TEBAKA:

    <pasta>/
        info.json            quem gerou, como, e a lista das imagens
        test/0000.npz        copas previstas para a imagem 0 do teste
        test/0001.npz        (ou test/0001.png, PNG de 16 bits)
        ...

Cada arquivo é uma imagem de etiquetas do tamanho do recorte (256x256):
0 = fundo, 1..N = uma copa cada. No .npz, o array fica na chave
"instances". O número do arquivo é a posição da imagem na divisão, na ordem
do conjunto AgML (a mesma que rsskd.data.AgmlTreeCrowns usa).

Para não haver dúvida sobre a ordem, `split_manifest` lista, para cada
posição, uma soma de verificação (md5) das bandas da imagem. O info.json de
uma pasta exportada leva essa lista, e a avaliação a confere.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .pseudolabels import PseudoLabelStore

INFO_FILE = "info.json"


def bands_md5(bands: np.ndarray) -> str:
    """Soma de verificação curta das bandas de uma imagem (12 caracteres)."""
    return hashlib.md5(np.ascontiguousarray(bands).tobytes()).hexdigest()[:12]


def split_manifest(dataset) -> list[dict]:
    """Lista [{indice, md5_bandas}] das imagens de uma divisão (AgmlTreeCrowns)."""
    return [{"indice": i, "md5_bandas": bands_md5(dataset[i].bands)}
            for i in range(len(dataset))]


class PredictionStore(PseudoLabelStore):
    """Uma pasta de previsões. Mesma organização das copas do professor,
    com duas diferenças: lê também PNG de 16 bits, e guarda um info.json
    em vez do parametros.json do professor.

    Uso:
        store = PredictionStore("previsoes/felipe_sam")
        store.load("test", 0)            # (H, W) int32, 0..N
        store.indices("test")            # posições com previsão
    """

    def __init__(self, root: str | Path):
        super().__init__(root)          # sem professor: não grava parametros.json

    def _png(self, split: str, index: int) -> Path:
        return self.root / split / f"{index:04d}.png"

    def exists(self, split: str, index: int) -> bool:
        return super().exists(split, index) or self._png(split, index).exists()

    def load(self, split: str, index: int) -> np.ndarray:
        if super().exists(split, index):
            return super().load(split, index)
        from PIL import Image

        with Image.open(self._png(split, index)) as im:
            return np.asarray(im).astype(np.int32)

    def indices(self, split: str) -> list[int]:
        folder = self.root / split
        if not folder.exists():
            return []
        return sorted({int(p.stem) for p in folder.iterdir()
                       if p.suffix in (".npz", ".png") and p.stem.isdigit()})

    def write_info(self, **info) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        info = {"criado_em": datetime.now().isoformat(timespec="seconds")} | info
        (self.root / INFO_FILE).write_text(json.dumps(info, indent=2, ensure_ascii=False),
                                           encoding="utf-8")

    def info(self) -> dict:
        path = self.root / INFO_FILE
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def check_against(store, split: str, dataset) -> list[str]:
    """Confere uma pasta de previsões com a divisão do conjunto.

    Devolve a lista de problemas (vazia = tudo certo): imagens sem previsão,
    previsões a mais, tamanho errado, ou bandas diferentes das listadas no
    info.json (ordem trocada).
    """
    problems = []
    n = len(dataset)
    have = set(store.indices(split))
    missing = sorted(set(range(n)) - have)
    extra = sorted(have - set(range(n)))
    if missing:
        problems.append(f"{len(missing)} imagens sem previsão (ex.: {missing[:5]})")
    if extra:
        problems.append(f"{len(extra)} previsões além das {n} imagens (ex.: {extra[:5]})")
    manifest = store.info().get("imagens", {}).get(split) if hasattr(store, "info") else None
    for i in sorted(have & set(range(n))):
        s = dataset[i]
        pred = store.load(split, i)
        if pred.shape != s.mask.shape:
            problems.append(f"imagem {i}: previsão {pred.shape}, imagem {s.mask.shape}")
            break
        if manifest and i < len(manifest) and manifest[i]["md5_bandas"] != bands_md5(s.bands):
            problems.append(f"imagem {i}: bandas diferentes das do info.json (ordem trocada?)")
            break
    return problems
