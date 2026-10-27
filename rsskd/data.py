"""Leitura do conjunto de teste de copas (Hugging Face, Project-AgML).

Conjunto: Peng, Yan, Xu, Mo e Wei, "UAV-based two-stage deep learning for
tree crown segmentation and height estimation in Camellia oleifera
plantations", Ecological Informatics 96, 103868, 2026. Licença CC BY 4.0.
https://huggingface.co/datasets/Project-AgML/tree_crown_segmentation

Segundo a página do conjunto, cada linha dos arquivos Parquet tem:
    bands         as 5 bandas (formato [5, 256, 256], uint8)
    bands_shape   o formato acima
    bands_dtype   o tipo dos números ("uint8")
    band_metadata metadados em JSON (georreferência etc.)
    band_order    a ordem das bandas (vazia nesse conjunto)
    mask          a máscara de copa (anotação de referência)
    rgb           uma composição colorida pronta
    split         "train", "val" ou "test"

Não conseguimos inspecionar os arquivos antes de escrever este código (o
acesso ao Hugging Face está bloqueado no ambiente do assistente). Por isso,
a decodificação aceita os formatos mais comuns (bytes crus, listas e
imagens PNG) e o script scripts/inspect_agml.py mostra o formato real.

Este arquivo não usa PyTorch: devolve arrays NumPy. A conversão para
tensores fica para o laço de treino.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

REPO_ID = "Project-AgML/tree_crown_segmentation"
DEFAULT_DIR = Path("data") / "agml_tree_crown"


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------
def download_agml(local_dir: str | Path = DEFAULT_DIR) -> Path:
    """Baixa os arquivos Parquet do Hugging Face para local_dir (~330 MB).

    A pasta data/ é ignorada pelo git. Precisa de internet e da biblioteca
    huggingface_hub (está no requirements.txt).
    """
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=REPO_ID, repo_type="dataset",
                             local_dir=str(local_dir),
                             allow_patterns=["*.parquet", "README.md"])
    return Path(path)


def parquet_files(local_dir: str | Path = DEFAULT_DIR) -> list[Path]:
    files = sorted(Path(local_dir).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"Nenhum .parquet em {local_dir}. Rode antes: "
            "python scripts/inspect_agml.py --baixar"
        )
    return files


# --------------------------------------------------------------------------
# Decodificação de cada coluna
# --------------------------------------------------------------------------
def _decode_image_bytes(data: bytes) -> np.ndarray:
    """Decodifica uma imagem codificada (PNG, JPEG ou TIFF) para array."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        return np.array(im)


def decode_array(value, shape=None, dtype=None) -> np.ndarray:
    """Converte o valor de uma célula do Parquet num array NumPy.

    Formatos aceitos:
      - dicionário de imagem do Hugging Face: {"bytes": ..., "path": ...};
      - bytes crus de um array (usa shape e dtype para remontar);
      - bytes de uma imagem codificada (PNG/JPEG/TIFF);
      - listas (aninhadas ou não) de números.
    """
    if isinstance(value, dict) and "bytes" in value:
        value = value["bytes"]

    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        # Imagens codificadas começam com assinaturas conhecidas.
        if data[:8] == b"\x89PNG\r\n\x1a\n" or data[:2] == b"\xff\xd8" \
                or data[:4] in (b"II*\x00", b"MM\x00*"):
            return _decode_image_bytes(data)
        if shape is None:
            raise ValueError("Bytes crus sem 'shape' para remontar o array.")
        arr = np.frombuffer(data, dtype=np.dtype(dtype or "uint8"))
        return arr.reshape(tuple(shape))

    arr = np.asarray(value if not hasattr(value, "tolist") else value.tolist())
    if arr.dtype == object:  # listas aninhadas de comprimento variável
        arr = np.stack([np.asarray(v) for v in value])
    if shape is not None and arr.size == int(np.prod(shape)):
        arr = arr.reshape(tuple(shape))
    if dtype is not None:
        arr = arr.astype(np.dtype(dtype))
    return arr


def mask_to_binary(mask: np.ndarray) -> np.ndarray:
    """Máscara de referência como 0/1, mesmo que venha em 0/255 ou RGB."""
    if mask.ndim == 3:
        mask = mask.max(axis=-1)
    return (mask > 0).astype(np.uint8)


def mask_to_instances(mask: np.ndarray) -> np.ndarray:
    """Numera as copas da referência pelas manchas desconectadas.

    As máscaras do conjunto são de copa contra fundo (sem número por árvore).
    Como as copas da plantação são quase todas separadas, cada mancha
    desconectada é tratada como uma árvore. Copas encostadas na referência
    viram uma só; isso deve ser lembrado na avaliação.
    """
    labels, _ = ndi.label(mask_to_binary(mask))
    return labels.astype(np.int32)


# --------------------------------------------------------------------------
# Conjunto de dados
# --------------------------------------------------------------------------
@dataclass
class Sample:
    bands: np.ndarray       # (C, H, W), como vem no arquivo
    mask: np.ndarray        # (H, W), 0/1
    rgb: np.ndarray | None  # (H, W, 3) ou None
    split: str
    index: int
    metadata: dict


class AgmlTreeCrowns:
    """Acesso às amostras do conjunto, com filtro por divisão.

    Uso:
        ds = AgmlTreeCrowns(split="train")
        amostra =y ds[0]
        amostra.bands.shape   # (5, 256, 256)
        amostra.mask.shape    # (256, 256)
    """

    def __init__(self, local_dir: str | Path = DEFAULT_DIR,
                 split: str | None = None):
        import pandas as pd

        frames = [pd.read_parquet(f) for f in parquet_files(local_dir)]
        df = pd.concat(frames, ignore_index=True)
        if split is not None:
            if "split" not in df.columns:
                raise KeyError("O arquivo não tem a coluna 'split'.")
            df = df[df["split"] == split].reset_index(drop=True)
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> Sample:
        row = self.df.iloc[i]
        shape = row.get("bands_shape")
        shape = None if shape is None else [int(s) for s in shape]
        bands = decode_array(row["bands"], shape, row.get("bands_dtype"))

        mask = mask_to_binary(decode_array(row["mask"]))
        rgb = decode_array(row["rgb"]) if "rgb" in row and row["rgb"] is not None else None

        meta = row.get("band_metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {"raw": meta}
        return Sample(bands=bands, mask=mask, rgb=rgb,
                      split=str(row.get("split", "")), index=i,
                      metadata=meta or {})


# --------------------------------------------------------------------------
# Descobrir a ordem das bandas pelos dados
# --------------------------------------------------------------------------
def band_contrast(dataset, n: int = 50) -> np.ndarray:
    """Razão média (copa / fundo) de cada banda, nas n primeiras amostras.

    Em vegetação viva, o NIR é muito mais alto na copa do que no solo
    (razão bem acima de 1), e o vermelho é mais baixo (razão abaixo de 1).
    Assim, a banda de maior razão é a candidata a NIR, e a de menor razão,
    a candidata a vermelho. É uma pista, não uma prova: confira com o NDVI.
    """
    ratios = []
    for i in range(min(n, len(dataset))):
        s = dataset[i]
        m = s.mask.astype(bool)
        if m.all() or not m.any():
            continue
        b = s.bands.astype(np.float32)
        ratios.append(b[:, m].mean(axis=1) / (b[:, ~m].mean(axis=1) + 1e-6))
    return np.mean(ratios, axis=0)


def guess_nir_red(dataset, n: int = 50) -> tuple[int, int, np.ndarray]:
    """Índices prováveis do NIR (maior razão) e do vermelho (menor razão)."""
    r = band_contrast(dataset, n)
    return int(np.argmax(r)), int(np.argmin(r)), r


# --------------------------------------------------------------------------
# Avaliação simples do professor contra a referência
# --------------------------------------------------------------------------
def iou_binary(pred: np.ndarray, ref: np.ndarray) -> float:
    """IoU entre duas máscaras 0/1 (copa contra fundo)."""
    p, r = pred.astype(bool), ref.astype(bool)
    union = (p | r).sum()
    return float((p & r).sum() / union) if union else 1.0


def match_instances(pred: np.ndarray, ref: np.ndarray,
                    iou_thr: float = 0.5) -> dict:
    """Casa copas previstas com copas da referência (IoU >= iou_thr).

    É o critério do Harmon et al. (2022): uma copa prevista é acerto se tem
    IoU >= 0,5 com uma copa da referência; cada copa da referência só pode
    ser casada uma vez. Retorna verdadeiros positivos (tp), falsos positivos
    (fp), falsos negativos (fn), precisão, revocação, F1 e o IoU médio dos
    acertos.
    """
    pred_ids = [i for i in np.unique(pred) if i > 0]
    ref_ids = [i for i in np.unique(ref) if i > 0]

    # Tabela de IoU entre todas as copas previstas e de referência.
    ious = np.zeros((len(pred_ids), len(ref_ids)))
    for a, pid in enumerate(pred_ids):
        pm = pred == pid
        # Só compara com as copas de referência que tocam esta copa prevista.
        for rid in np.unique(ref[pm]):
            if rid == 0:
                continue
            b = ref_ids.index(rid)
            rm = ref == rid
            ious[a, b] = (pm & rm).sum() / (pm | rm).sum()

    # Casamento guloso: maiores IoU primeiro.
    tp, matched_ious = 0, []
    used_p, used_r = set(), set()
    for a, b in sorted(np.argwhere(ious >= iou_thr).tolist(),
                       key=lambda ab: -ious[ab[0], ab[1]]):
        if a in used_p or b in used_r:
            continue
        used_p.add(a)
        used_r.add(b)
        tp += 1
        matched_ious.append(ious[a, b])

    fp, fn = len(pred_ids) - tp, len(ref_ids) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec,
            "f1": f1, "mean_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
            "n_pred": len(pred_ids), "n_ref": len(ref_ids)}
