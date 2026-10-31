"""Aumentos de dados do aluno e montagem dos lotes.

É a terceira peça da etapa 1 do aluno.

**Quais transformações.** Uma imagem de drone é tirada de cima: não existe
"lado de cima" nem "esquerda". Girar a imagem 90°, 180° ou 270°, ou
espelhá-la, produz outra imagem perfeitamente plausível do mesmo pomar.
Essas 8 combinações (4 rotações x com ou sem espelhamento; em matemática,
o grupo diedral D4) têm duas vantagens:

  - são exatas: só trocam pixels de lugar, sem interpolação, então as
    máscaras continuam binárias e as copas não mudam de área;
  - não mudam o tamanho das copas, o que importa para a regra de área
    média de Harmon et al. na etapa 2.

**Para que servem.** Duas coisas, em momentos diferentes:

  1. Aumento de dados (agora, na etapa 1): a cada época, cada imagem é vista
     numa orientação sorteada. Com 476 imagens de treino, isso equivale a
     até 8 x 476 imagens diferentes e reduz o sobreajuste.
  2. Consistência a transformações do RS-SSKD (item 4): o aluno prevê as
     copas na imagem original e na transformada; desfazendo a transformação
     da segunda previsão, as duas devem coincidir. Para isso servem as
     funções em torch e a inversa (`inverse_dihedral`), já testadas aqui.

O RS-SSKD original usa espelhamento e mudança de escala. A escala fica de
fora por enquanto, porque muda a área das copas; pode entrar depois como
ablação.

**Lotes.** Cada imagem tem um número diferente de copas (N), então os alvos
não podem ser empilhados num tensor só. O Mask R-CNN recebe uma lista de
imagens e uma lista de alvos; `collate` monta essas listas.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# As 8 transformações, como pares (k, espelha): rotação de k x 90° no
# sentido anti-horário, seguida (ou não) de espelhamento horizontal.
DIHEDRAL = [(k, flip) for flip in (False, True) for k in range(4)]


# --------------------------------------------------------------------------
# Transformações (NumPy, para os dados; torch, para as previsões do item 4)
# --------------------------------------------------------------------------
def dihedral(x: np.ndarray, k: int, flip: bool) -> np.ndarray:
    """Aplica a transformação (k, flip) nos dois últimos eixos (H, W).

    Serve tanto para as bandas (C, H, W) quanto para as copas (H, W).
    """
    x = np.rot90(x, k, axes=(-2, -1))
    return x[..., ::-1] if flip else x


def dihedral_torch(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    """O mesmo que `dihedral`, para tensores (ex.: máscaras previstas)."""
    x = torch.rot90(x, k, dims=(-2, -1))
    return x.flip(-1) if flip else x


def inverse_dihedral(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    """Desfaz `dihedral_torch(x, k, flip)`: primeiro desespelha, depois gira
    de volta (-k x 90°). É o passo que leva a previsão feita na imagem
    transformada de volta à orientação original."""
    if flip:
        x = x.flip(-1)
    return torch.rot90(x, -k, dims=(-2, -1))


class RandomDihedral:
    """Sorteia uma das 8 transformações e aplica nas bandas e nas copas.

    É a `transform` do CrownDataset (parte 2b):
        ds = CrownDataset("train", ..., transform=RandomDihedral())

    O sorteio usa o gerador do PyTorch, que o DataLoader inicializa com uma
    semente diferente em cada processo; assim, com num_workers > 1, os
    processos não repetem as mesmas orientações. Para reproduzir um treino,
    basta fixar torch.manual_seed no início.
    """

    def __call__(self, bands: np.ndarray, crowns: np.ndarray):
        k, flip = DIHEDRAL[int(torch.randint(len(DIHEDRAL), ()).item())]
        # A mesma transformação nas duas partes: as copas continuam em cima
        # das árvores da imagem.
        return dihedral(bands, k, flip), dihedral(crowns, k, flip)


# --------------------------------------------------------------------------
# Lotes
# --------------------------------------------------------------------------
def collate(batch):
    """Junta [(imagem, alvo), ...] em (lista de imagens, lista de alvos).

    O comportamento padrão do PyTorch tentaria empilhar as caixas de todas
    as imagens num tensor só, o que falha quando o número de copas muda.
    """
    images, targets = zip(*batch)
    return list(images), list(targets)


def make_loader(dataset: Dataset, batch_size: int = 4, shuffle: bool = True,
                num_workers: int = 0) -> DataLoader:
    """DataLoader do aluno, com a montagem de lotes certa.

    num_workers: processos que preparam os lotes em paralelo com a GPU. No
    Windows, valores > 0 só funcionam com o código dentro de
    `if __name__ == "__main__":`; por isso o padrão é 0.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, collate_fn=collate)
