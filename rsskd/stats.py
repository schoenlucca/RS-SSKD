"""Incerteza das métricas: intervalos de confiança e comparação pareada.

As sementes mostram se o *treino* é estável; não dizem se a diferença entre
dois métodos se mantém em *outras imagens*. Com 59 ou 60 imagens, uma
diferença de 0,01 no F1 pode vir de duas ou três imagens favoráveis. O
bootstrap mede isso sem supor nenhuma distribuição:

  - sorteia, com reposição, n imagens das n avaliadas (algumas repetem,
    outras ficam de fora) e recalcula a métrica com elas;
  - repete milhares de vezes; os percentis 2,5% e 97,5% dos valores
    formam o intervalo de confiança de 95%.

A **unidade sorteada é a imagem**, não a copa: copas da mesma imagem não são
independentes (mesma iluminação, mesma densidade), e sortear copas daria um
intervalo estreito demais.

Para comparar dois métodos nas mesmas imagens, o bootstrap é **pareado**: o
mesmo sorteio de imagens vale para os dois, e o que se mede é a diferença.
Assim, uma imagem difícil para os dois não aumenta a incerteza da
comparação. O valor p (bilateral) é o dobro da fração de sorteios em que a
diferença tem o sinal oposto ao observado.

As métricas vêm das contagens por imagem que rsskd.evaluation.CrownMetrics
grava em .rows (tp, fp, fn, n_pred, n_ref). O F1 é o micro, somando as
contagens das imagens sorteadas, como no resumo de CrownMetrics.

Referência: EFRON, B.; TIBSHIRANI, R. An Introduction to the Bootstrap.
Chapman & Hall, 1993.
"""

from __future__ import annotations

import numpy as np

METRICS = ("f1", "precisao", "revocacao", "erro_contagem")
COLUMNS = ("tp", "fp", "fn", "n_pred", "n_ref")


def counts_from_rows(rows: list[dict]) -> np.ndarray:
    """Matriz (imagens, 5) com tp, fp, fn, n_pred e n_ref de cada imagem."""
    return np.array([[float(r[c]) for c in COLUMNS] for r in rows], dtype=float)


def metric_from_counts(counts: np.ndarray, metric: str = "f1") -> np.ndarray:
    """A métrica para um conjunto de imagens, ou para vários de uma vez.

    counts: (imagens, 5) ou (sorteios, imagens, 5). Devolve um escalar ou um
    valor por sorteio.
    """
    if metric not in METRICS:
        raise ValueError(f"Métrica desconhecida: {metric}. Use uma de {METRICS}.")
    if metric == "erro_contagem":
        return np.abs(counts[..., 3] - counts[..., 4]).mean(axis=-1)
    tp, fp, fn = (counts[..., i].sum(axis=-1) for i in range(3))
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        if metric == "precisao":
            return prec
        if metric == "revocacao":
            return rec
        return np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)


def _draws(n_images: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_images, size=(n_boot, n_images))


def bootstrap_ci(rows: list[dict], metric: str = "f1", n_boot: int = 10000,
                 alpha: float = 0.05, seed: int = 0) -> dict:
    """Valor observado e intervalo de confiança de 1 - alpha (percentis).

    Devolve {"valor", "inf", "sup", "dp"} (dp = desvio do bootstrap).
    """
    counts = counts_from_rows(rows)
    if len(counts) < 2:
        raise ValueError("O bootstrap precisa de pelo menos 2 imagens.")
    value = float(metric_from_counts(counts, metric))
    boot = metric_from_counts(counts[_draws(len(counts), n_boot, seed)], metric)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"valor": value, "inf": float(lo), "sup": float(hi), "dp": float(boot.std())}


def paired_bootstrap(rows_a: list[dict], rows_b: list[dict], metric: str = "f1",
                     n_boot: int = 10000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Diferença A - B nas mesmas imagens, com intervalo e valor p.

    As duas listas precisam estar na mesma ordem de imagens (mesma divisão).
    Devolve {"diferenca", "inf", "sup", "p"}.
    """
    if len(rows_a) != len(rows_b):
        raise ValueError(f"Os métodos têm {len(rows_a)} e {len(rows_b)} imagens; "
                         "a comparação pareada precisa das mesmas.")
    ia = [r.get("indice") for r in rows_a]
    ib = [r.get("indice") for r in rows_b]
    if ia != ib:
        raise ValueError("As imagens dos dois métodos estão em ordens diferentes.")
    ca, cb = counts_from_rows(rows_a), counts_from_rows(rows_b)
    if len(ca) < 2:
        raise ValueError("O bootstrap precisa de pelo menos 2 imagens.")
    diff = float(metric_from_counts(ca, metric) - metric_from_counts(cb, metric))
    idx = _draws(len(ca), n_boot, seed)
    boot = metric_from_counts(ca[idx], metric) - metric_from_counts(cb[idx], metric)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    if diff == 0:
        p = 1.0
    else:
        opposite = np.mean(boot <= 0) if diff > 0 else np.mean(boot >= 0)
        p = float(min(1.0, 2 * opposite))
    return {"diferenca": diff, "inf": float(lo), "sup": float(hi), "p": p}


def format_ci(ci: dict, digits: int = 3) -> str:
    """'0,521 [0,489; 0,553]', com vírgula decimal, para tabelas."""
    def f(x):
        return f"{x:.{digits}f}".replace(".", ",")
    key = "valor" if "valor" in ci else "diferenca"
    sign = "+" if key == "diferenca" and ci[key] > 0 else ""
    return f"{sign}{f(ci[key])} [{f(ci['inf'])}; {f(ci['sup'])}]"
