"""Classificador few-shot por prototipos e amostrador de episodios."""

from __future__ import annotations

import random
from collections import defaultdict

import torch
import torch.nn.functional as F


def proto_logits(
    support_emb: torch.Tensor,
    support_y: torch.Tensor,
    query_emb: torch.Tensor,
    n_way: int,
    gamma: float = 10.0,
) -> torch.Tensor:
    """Eq. (8) e (9): prototipo = media do suporte; score = gamma * cosseno.

    support_y deve estar em {0, ..., n_way - 1}.
    """
    protos = torch.stack([support_emb[support_y == c].mean(0) for c in range(n_way)])
    q = F.normalize(query_emb, dim=1)
    p = F.normalize(protos, dim=1)
    return gamma * q @ p.t()


class EpisodeSampler:
    """Gera episodios N-way K-shot a partir de uma lista de rotulos.

    Cada episodio devolve (indices_suporte, indices_consulta, classes), com
    os indices referentes ao dataset original. Classes com menos de
    k_shot + n_query amostras sao ignoradas.
    """

    def __init__(self, labels, n_way: int, k_shot: int, n_query: int,
                 n_episodes: int, seed: int | None = None):
        self.n_way, self.k_shot, self.n_query = n_way, k_shot, n_query
        self.n_episodes = n_episodes
        self.rng = random.Random(seed)

        by_class = defaultdict(list)
        for idx, lab in enumerate(labels):
            by_class[int(lab)].append(idx)
        need = k_shot + n_query
        self.by_class = {c: v for c, v in by_class.items() if len(v) >= need}
        if len(self.by_class) < n_way:
            raise ValueError(
                f"Apenas {len(self.by_class)} classes tem >= {need} amostras; "
                f"sao necessarias {n_way}."
            )

    def __len__(self) -> int:
        return self.n_episodes

    def __iter__(self):
        for _ in range(self.n_episodes):
            classes = self.rng.sample(sorted(self.by_class), self.n_way)
            support, query = [], []
            for c in classes:
                picked = self.rng.sample(self.by_class[c], self.k_shot + self.n_query)
                support += picked[: self.k_shot]
                query += picked[self.k_shot:]
            yield support, query, classes
