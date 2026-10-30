"""Conjunto de dados do aluno, no formato que o Mask R-CNN do PyTorch lê.

É a segunda peça da etapa 1 do aluno. Junta três coisas que já existem:

    imagem (AgmlTreeCrowns)  --InputSpec-->  tensor (6, H, W) normalizado
    copas do professor (PseudoLabelStore)  -->  alvo do Mask R-CNN
    máscara de referência                   -->  guardada à parte

O que o Mask R-CNN espera de cada imagem (torchvision):

    imagem: tensor float32 (C, H, W)
    alvo:   {"boxes":  float32 (N, 4)  caixas [x_min, y_min, x_max, y_max],
             "labels": int64   (N,)    classe de cada copa (sempre 1),
             "masks":  uint8   (N, H, W) uma máscara binária por copa}

N é o número de copas daquela imagem e muda de imagem para imagem. Uma
imagem sem copas tem N = 0 (caixas (0, 4)); o Mask R-CNN aceita e aprende
com ela que ali não há árvore.

**Separação entre treino e avaliação.** O alvo vem sempre do professor:
é isso que substitui as anotações humanas. As máscaras de referência do
conjunto nunca entram no alvo; ficam disponíveis só por reference(i), que
o laço de avaliação usa para medir o aluno. Assim não há como o aluno
"ver" a resposta durante o treino.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import DEFAULT_DIR, AgmlTreeCrowns, mask_to_instances
from .inputs import InputSpec
from .pseudolabels import PseudoLabelStore
from .teacher import instances_to_targets

# Uma transformação recebe (bandas (C, H, W), copas (H, W)) e devolve o par
# transformado. Os aumentos de dados da parte 2c seguem esse formato: girar
# a imagem de etiquetas junto com as bandas mantém as copas alinhadas.
Transform = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


class CrownDataset(Dataset):
    """Imagens do AgML com as copas do professor como alvo.

    Uso:
        spec = InputSpec.load("configs/agml_entrada.json")
        ds = CrownDataset(split="train", pseudo_dir="data/pseudo/agml", spec=spec)
        imagem, alvo = ds[0]
        imagem.shape            # torch.Size([6, 256, 256])
        alvo["boxes"].shape     # torch.Size([N, 4])
        ref = ds.reference(0)   # copas da referência, só para avaliar
    """

    def __init__(self, split: str, pseudo_dir: str | Path, spec: InputSpec,
                 dataset_dir: str | Path = DEFAULT_DIR,
                 transform: Transform | None = None):
        self.split = split
        self.spec = spec
        self.transform = transform
        self.images = AgmlTreeCrowns(dataset_dir, split=split)
        self.store = PseudoLabelStore(pseudo_dir)

        # Confere logo no início que toda imagem tem copas do professor, em
        # vez de descobrir no meio de um treino de horas.
        missing = sorted(set(range(len(self.images))) - set(self.store.indices(split)))
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} imagens de '{split}' sem copas do professor "
                f"em {pseudo_dir} (ex.: {missing[:5]}). Rode antes: "
                f"python scripts/generate_pseudolabels.py")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, dict]:
        bands = self.images[i].bands
        crowns = self.store.load(self.split, i)        # (H, W), 0..N
        if self.transform is not None:
            bands, crowns = self.transform(bands, crowns)

        # Entrada: 5 bandas + NDVI, normalizadas (parte 2a).
        image = torch.from_numpy(np.ascontiguousarray(self.spec(bands)))

        # Alvo: uma máscara e uma caixa por copa do professor.
        t = instances_to_targets(crowns)
        target = {
            "boxes": torch.from_numpy(t["boxes"]).reshape(-1, 4),
            "labels": torch.from_numpy(t["labels"]),
            "masks": torch.from_numpy(t["masks"]),
            "image_id": torch.tensor(i),
        }
        return image, target

    def reference(self, i: int) -> np.ndarray:
        """Copas da referência (0 = fundo, 1..N), só para a avaliação."""
        return mask_to_instances(self.images[i].mask)
