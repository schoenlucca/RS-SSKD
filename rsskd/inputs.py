"""Entrada do aluno: as bandas da imagem mais o NDVI, normalizadas.

É a primeira peça da etapa 1 do aluno. O professor TEBAKA olha só para o
NDVI; o aluno (Mask R-CNN com ResNet-101) recebe tudo:

    banda 0 .. banda 4  (vermelho, verde, azul, NIR, Red-Edge no AgML)
    + NDVI              (calculado aqui, a partir do NIR e do vermelho)
    = 6 canais

Por que acrescentar o NDVI, se a rede já tem o NIR e o vermelho? Em
princípio ela poderia aprender a combinação sozinha, mas uma divisão
(NIR - V) / (NIR + V) é difícil de imitar com somas e multiplicações de
uma camada convolucional. Dar o NDVI pronto é uma "dica" barata, e é o
canal em que o professor se baseou.

Por que normalizar? Cada canal tem a sua escala: as bandas vão de 0 a 255 e
o NDVI de -1 a 1. Sem normalizar, os canais de valores grandes dominam o
início do treino e os pesos da primeira camada ficam desequilibrados. A
normalização usual é, para cada canal c:

    x_c' = (x_c - media_c) / desvio_c

com a média e o desvio calculados **só nas imagens de treino** (usar as de
validação ou teste seria deixar o modelo "espiar" dados de avaliação). Os
valores ficam em configs/agml_entrada.json, gerado por
scripts/compute_input_stats.py.

Tudo aqui é NumPy; a conversão para tensor do PyTorch fica no Dataset
(parte 2b).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .teacher import compute_ndvi


def build_input(bands: np.ndarray, nir_band: int, red_band: int,
                add_ndvi: bool = True) -> np.ndarray:
    """Empilha as bandas e o NDVI numa imagem (C, H, W) em float32.

    bands: (C, H, W), como vem do conjunto (AgmlTreeCrowns).
    Retorna (C + 1, H, W) com o NDVI como último canal, ou (C, H, W) se
    add_ndvi=False. Os valores ainda não estão normalizados.
    """
    x = bands.astype(np.float32)
    if not add_ndvi:
        return x
    ndvi = compute_ndvi(x[nir_band], x[red_band])
    return np.concatenate([x, ndvi[None]], axis=0)


class ChannelStats:
    """Média e desvio de cada canal, acumulados imagem por imagem.

    Não dá para empilhar as 476 imagens de treino na memória só para tirar a
    média, e nem é preciso: basta somar, para cada canal, os valores (S1),
    os quadrados (S2) e a quantidade de pixels (n). No fim,

        media  = S1 / n
        desvio = sqrt(S2 / n - media^2)

    As somas usam float64 para não perder precisão com milhões de pixels.
    """

    def __init__(self):
        # Começam vazios; o tamanho (número de canais) vem da 1a imagem.
        self.s1 = np.zeros(0)
        self.s2 = np.zeros(0)
        self.n = 0

    def update(self, x: np.ndarray) -> None:
        """x: (C, H, W)."""
        flat = x.reshape(x.shape[0], -1).astype(np.float64)
        if self.n == 0:
            self.s1 = np.zeros(flat.shape[0])
            self.s2 = np.zeros(flat.shape[0])
        self.s1 += flat.sum(axis=1)
        self.s2 += (flat ** 2).sum(axis=1)
        self.n += flat.shape[1]

    @property
    def mean(self) -> np.ndarray:
        return self.s1 / self.n

    @property
    def std(self) -> np.ndarray:
        var = self.s2 / self.n - self.mean ** 2
        return np.sqrt(np.maximum(var, 0.0))


@dataclass
class InputSpec:
    """Receita da entrada do aluno: quais bandas, se tem NDVI, e a
    normalização. Chamar a receita numa imagem devolve a entrada pronta.

    Uso:
        spec = InputSpec.load("configs/agml_entrada.json")
        x = spec(amostra.bands)        # (6, 256, 256), normalizada
        spec.in_channels               # 6, para criar a primeira camada
    """

    nir_band: int
    red_band: int
    add_ndvi: bool = True
    channels: list[str] = field(default_factory=list)   # nomes, para o relatório
    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)

    @property
    def in_channels(self) -> int:
        return len(self.mean)

    def __call__(self, bands: np.ndarray) -> np.ndarray:
        x = build_input(bands, self.nir_band, self.red_band, self.add_ndvi)
        if x.shape[0] != self.in_channels:
            raise ValueError(f"A imagem gerou {x.shape[0]} canais, mas a "
                             f"normalização tem {self.in_channels}.")
        mean = np.asarray(self.mean, np.float32)[:, None, None]
        # Um canal constante teria desvio 0; o piso evita dividir por zero.
        std = np.maximum(np.asarray(self.std, np.float32), 1e-6)[:, None, None]
        return (x - mean) / std

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Desfaz a normalização: x_c = x_c' * desvio_c + media_c.

        Serve para desenhar a entrada do aluno com as cores originais
        (scripts/show_batch.py). x: (C, H, W), já normalizada.
        """
        mean = np.asarray(self.mean, np.float32)[:, None, None]
        std = np.maximum(np.asarray(self.std, np.float32), 1e-6)[:, None, None]
        return x * std + mean

    def save(self, path: str | Path, comment: str = "") -> None:
        info = {"_comentario": comment} if comment else {}
        info.update(asdict(self))
        Path(path).write_text(json.dumps(info, indent=2, ensure_ascii=False),
                              encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "InputSpec":
        info = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in info.items() if not k.startswith("_")})


def compute_input_spec(bands_iter, nir_band: int, red_band: int,
                       channels: list[str], add_ndvi: bool = True) -> InputSpec:
    """Calcula a média e o desvio de cada canal sobre um conjunto de imagens.

    bands_iter: imagens (C, H, W) do treino, uma de cada vez.
    channels:   nomes dos canais na ordem da entrada (com "ndvi" no fim).
    """
    stats = ChannelStats()
    for bands in bands_iter:
        stats.update(build_input(bands, nir_band, red_band, add_ndvi))
    if stats.n == 0:
        raise ValueError("Nenhuma imagem para calcular a normalização.")
    if len(channels) != len(stats.mean):
        raise ValueError(f"{len(channels)} nomes para {len(stats.mean)} canais.")
    return InputSpec(nir_band, red_band, add_ndvi, list(channels),
                     [round(float(v), 6) for v in stats.mean],
                     [round(float(v), 6) for v in stats.std])
