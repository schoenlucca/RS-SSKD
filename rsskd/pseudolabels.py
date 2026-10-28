"""Guarda e lê as copas do professor TEBAKA (os pseudo-rótulos) no disco.

O professor leva alguns segundos por imagem. O aluno vai passar pelas mesmas
imagens dezenas de vezes (uma por época), então calculamos as copas uma vez
e guardamos. É o mesmo papel que um arquivo de anotações teria num conjunto
rotulado à mão, só que gerado pelo professor.

Organização no disco:

    <raiz>/
        parametros.json      parâmetros do professor que gerou as copas
        train/0000.npz       copas da amostra 0 do treino
        train/0001.npz
        ...
        val/...
        test/...

Cada .npz guarda uma imagem de etiquetas (0 = fundo, 1..N = copa 1..N),
comprimida. O parametros.json garante que copas geradas com parâmetros
diferentes não se misturem: se o professor mudar, o armazenamento recusa
gravar por cima e pede uma pasta nova (ou a remoção da antiga).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

PARAMS_FILE = "parametros.json"


class PseudoLabelStore:
    """Armazenamento das copas do professor, por divisão e índice.

    Uso:
        store = PseudoLabelStore("data/pseudo/agml", teacher)
        store.save("train", 0, copas)       # copas: (H, W), 0..N
        copas = store.load("train", 0)
        store.indices("train")              # [0, 1, 2, ...] já gerados
    """

    def __init__(self, root: str | Path, teacher=None):
        self.root = Path(root)
        self.params_path = self.root / PARAMS_FILE
        if teacher is not None:
            self._check_or_write_params(teacher)

    # ------------------------------------------------------------------
    # Parâmetros do professor
    # ------------------------------------------------------------------
    @staticmethod
    def _teacher_params(teacher) -> dict:
        """Parâmetros do professor como dicionário simples (para o JSON)."""
        params = asdict(teacher)
        params.pop("extra", None)
        return params

    def _check_or_write_params(self, teacher) -> None:
        """Grava os parâmetros na primeira vez; depois, só confere.

        Se a pasta já tem copas geradas com outros parâmetros, levanta um
        erro. Misturar copas de professores diferentes no mesmo treino
        tornaria o resultado impossível de reproduzir.
        """
        current = self._teacher_params(teacher)
        if self.params_path.exists():
            saved = self.params()["professor"]
            if saved != current:
                diff = {k: (saved.get(k), current.get(k))
                        for k in set(saved) | set(current)
                        if saved.get(k) != current.get(k)}
                raise ValueError(
                    f"As copas em {self.root} foram geradas com outros "
                    f"parâmetros (salvo, atual): {diff}. Use outra pasta ou "
                    f"apague a antiga."
                )
            return
        self.root.mkdir(parents=True, exist_ok=True)
        info = {"professor": current,
                "criado_em": datetime.now().isoformat(timespec="seconds")}
        self.params_path.write_text(json.dumps(info, indent=2, ensure_ascii=False),
                                    encoding="utf-8")

    def params(self) -> dict:
        """Lê o parametros.json (professor e data de criação)."""
        return json.loads(self.params_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Copas
    # ------------------------------------------------------------------
    def path(self, split: str, index: int) -> Path:
        return self.root / split / f"{index:04d}.npz"

    def exists(self, split: str, index: int) -> bool:
        return self.path(split, index).exists()

    def save(self, split: str, index: int, instances: np.ndarray) -> None:
        """Grava as copas de uma imagem.

        As etiquetas são guardadas como uint16 (até 65.535 copas por
        imagem, muito mais que o necessário), com compressão. Uma imagem de
        256x256 ocupa poucos kB, porque as etiquetas são quase todas iguais
        a seus vizinhos.
        """
        if instances.max() > np.iinfo(np.uint16).max:
            raise ValueError("Copas demais para uint16.")
        p = self.path(split, index)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, instances=instances.astype(np.uint16))

    def load(self, split: str, index: int) -> np.ndarray:
        """Lê as copas de uma imagem como int32 (0 = fundo, 1..N = copas)."""
        with np.load(self.path(split, index)) as f:
            return f["instances"].astype(np.int32)

    def indices(self, split: str) -> list[int]:
        """Índices das imagens que já têm copas gravadas nesta divisão."""
        folder = self.root / split
        if not folder.exists():
            return []
        return sorted(int(p.stem) for p in folder.glob("*.npz"))
