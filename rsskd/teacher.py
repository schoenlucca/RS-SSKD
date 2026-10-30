"""Professor TEBAKA: gera as copas (pseudo-anotações) sem nenhum rótulo humano.

Baseado em Epifani, D'Avino e Caruso, "TEBAKA: Territorial Basic Knowledge
Acquisition. An Agritech Project for Italy: Results on Self-Supervised
Semantic Segmentation", IEEE ISCC 2023, p. 1116-1121.

O que o TEBAKA faz (Fig. 3 do artigo) e o que acrescentamos:

  1. NDVI = (NIR - Vermelho) / (NIR + Vermelho), pixel a pixel.          [TEBAKA]
  2. Limiar th1, erosão 5x5 e dilatação 5x5 (remove pontos soltos).     [TEBAKA]
  3. Remoção de ruído FastNLMeans e novo limiar th2.                     [TEBAKA]
  4. Erosão 5x5, dilatação 3x3, erosão 3x3, dilatação 3x3 (acabamento).  [TEBAKA]
  5. Separação em copas individuais (watershed sobre a distância ao     [nosso]
     fundo), porque queremos uma máscara por árvore, não só
     "vegetação contra fundo".
  6. Filtro por área (descarta manchas pequenas demais).                 [nosso]

O resultado do passo 5/6 substitui as anotações humanas no treino do aluno.

Os valores padrão dos parâmetros são pontos de partida: o artigo do TEBAKA
não publica os valores que usou, e eles devem ser ajustados olhando as
imagens do Cerrado (ver scripts/visualize_teacher.py).

Este arquivo usa só NumPy, SciPy, scikit-image e OpenCV (sem PyTorch).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed


# --------------------------------------------------------------------------
# Passo 1: NDVI
# --------------------------------------------------------------------------
def compute_ndvi(nir: np.ndarray, red: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """NDVI = (NIR - Vermelho) / (NIR + Vermelho), com valores em [-1, 1].

    eps evita divisão por zero em pixels sem sinal (bordas pretas do
    ortomosaico, por exemplo).
    """
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    return (nir - red) / (nir + red + eps)


# --------------------------------------------------------------------------
# Operações morfológicas
# --------------------------------------------------------------------------
def _kernel(size: int) -> np.ndarray:
    """Elemento estruturante quadrado de 1s (como no TEBAKA)."""
    return np.ones((size, size), np.uint8)


def _erode(mask: np.ndarray, size: int) -> np.ndarray:
    # Erosão: "come" as bordas das manchas brancas; apaga pontos isolados.
    return cv2.erode(mask, _kernel(size))


def _dilate(mask: np.ndarray, size: int) -> np.ndarray:
    # Dilatação: "engorda" as manchas; devolve o tamanho após a erosão.
    return cv2.dilate(mask, _kernel(size))


# --------------------------------------------------------------------------
# Passos 2 a 4: máscara de vegetação (o "Tunable Denoising Module")
# --------------------------------------------------------------------------
def vegetation_mask(
    ndvi: np.ndarray,
    th1: float | str = 0.4,
    th2: float | str = 0.4,
    nlm_h: float = 10.0,
    nlm_template: int = 7,
    nlm_search: int = 21,
) -> np.ndarray:
    """Máscara binária (0 = fundo, 1 = vegetação) a partir do NDVI.

    Parâmetros ajustáveis (os 5 do TEBAKA):
        th1, th2:     limiares de NDVI antes e depois da remoção de ruído.
        nlm_h:        força da remoção de ruído (str no artigo).
        nlm_template: tamanho da janela de comparação (tw no artigo).
        nlm_search:   tamanho da janela de busca (sw no artigo).

    Interpretação nossa de um ponto que o artigo não detalha: a remoção de
    ruído é aplicada aos valores de NDVI dentro da máscara do primeiro
    limiar (convertidos para 0-255, formato exigido pelo OpenCV), e o
    segundo limiar é aplicado ao resultado.

    th1 e th2 também aceitam "otsu": o limiar é escolhido automaticamente em
    cada imagem pelo método de Otsu (o valor que melhor separa os pixels em
    dois grupos). Útil quando o NDVI não está calibrado (bandas em 8 bits,
    sem reflectância) e os valores mudam de uma imagem para outra.
    """
    otsu = float(threshold_otsu(ndvi)) if "otsu" in (th1, th2) else 0.0
    th1 = otsu if th1 == "otsu" else float(th1)
    th2 = otsu if th2 == "otsu" else float(th2)

    # Passo 2: limiar th1, depois abertura (erosão seguida de dilatação) 5x5.
    m = (ndvi > th1).astype(np.uint8)
    m = _dilate(_erode(m, 5), 5)

    # Passo 3: remoção de ruído e segundo limiar.
    # Leva o NDVI de [-1, 1] para [0, 255] só dentro da máscara.
    ndvi_u8 = np.clip((ndvi + 1.0) * 127.5, 0, 255).astype(np.uint8) * m
    den = cv2.fastNlMeansDenoising(ndvi_u8, None, nlm_h, nlm_template, nlm_search)
    th2_u8 = (th2 + 1.0) * 127.5
    m = (den > th2_u8).astype(np.uint8)

    # Passo 4: acabamento, na sequência do artigo.
    m = _dilate(_erode(m, 5), 3)
    m = _dilate(_erode(m, 3), 3)
    return m


# --------------------------------------------------------------------------
# Passo 5: separação em copas individuais
# --------------------------------------------------------------------------
def split_crowns(
    mask: np.ndarray,
    min_distance: int = 15,
    min_area: int = 200,
) -> np.ndarray:
    """Separa a máscara de vegetação em copas, uma etiqueta por árvore.

    Ideia: calcula, para cada pixel de vegetação, a distância até o fundo.
    O centro de uma copa fica longe do fundo (pico da distância); o ponto
    onde duas copas se encostam fica perto (vale). O watershed "inunda" a
    partir dos picos e traça a fronteira nos vales, separando as copas.

    Parâmetros:
        min_distance: distância mínima (em pixels) entre dois centros de
                      copa. Valores pequenos dividem copas grandes em várias;
                      valores grandes juntam copas vizinhas. Deve ser
                      próximo do raio de uma copa pequena, em pixels.
        min_area:     manchas com menos pixels são descartadas (ruído, capim).

    Retorna uma imagem de etiquetas: 0 = fundo, 1..N = copa 1..N.
    """
    mask = mask.astype(bool)
    # Remove manchas pequenas antes de procurar os centros.
    comps = ndi.label(mask)[0]  # type: ignore[index]
    mask = _remove_small_labels(comps, min_area) > 0
    if not mask.any():
        return np.zeros(mask.shape, np.int32)

    dist = np.asarray(ndi.distance_transform_edt(mask))

    # Picos da distância = prováveis centros de copa.
    # labels= garante pelo menos um pico por mancha desconectada.
    comps = ndi.label(mask)[0]  # type: ignore[index]
    peaks = peak_local_max(dist, min_distance=min_distance, labels=comps,
                           exclude_border=False)
    markers = np.zeros(mask.shape, np.int32)
    peaks = np.asarray(peaks, np.intp)            # (n_picos, 2): linha, coluna
    markers[peaks[:, 0], peaks[:, 1]] = np.arange(1, len(peaks) + 1)

    # Watershed sobre -distância: os centros são os "fundos" dos vales.
    labels = watershed(-dist, markers, mask=mask)

    # Passo 6: remove copas pequenas que sobraram após a divisão.
    labels = _remove_small_labels(labels, min_area)
    return labels


def _remove_small_labels(labels: np.ndarray, min_area: int) -> np.ndarray:
    """Zera copas com área abaixo de min_area e renumera 1..N."""
    ids, counts = np.unique(labels[labels > 0], return_counts=True)
    keep = ids[counts >= min_area]
    out = np.zeros_like(labels)
    for new_id, old_id in enumerate(keep, start=1):
        out[labels == old_id] = new_id
    return out


# --------------------------------------------------------------------------
# Professor completo
# --------------------------------------------------------------------------
@dataclass
class TebakaTeacher:
    """Professor TEBAKA configurável.

    nir_band e red_band são os índices (começando em 0) das bandas NIR e
    vermelho na imagem. A ordem das bandas do conjunto de dados ainda não
    foi confirmada: ajuste aqui quando soubermos.

    Uso:
        teacher = TebakaTeacher(nir_band=3, red_band=1)
        out = teacher(imagem)          # imagem (H, W, C) ou (C, H, W)
        out["instances"]               # etiquetas 0..N, uma por copa
        out["n_crowns"]                # número de copas encontradas
    """

    nir_band: int = 3
    red_band: int = 1
    th1: float | str = 0.4
    th2: float | str = 0.4
    nlm_h: float = 10.0
    nlm_template: int = 7
    nlm_search: int = 21
    min_distance: int = 15
    min_area: int = 200
    channels_first: bool | None = None  # None = detectar automaticamente
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, path: str | Path) -> "TebakaTeacher":
        """Cria o professor a partir de um arquivo de configuração JSON.

        Lê a seção "professor" do arquivo (ver configs/agml_teacher.json).
        Chaves que começam com "_" são comentários e são ignoradas. Chaves
        desconhecidas geram erro, para um erro de digitação não passar
        despercebido.
        """
        cfg = load_config(path)["professor"]
        params = {k: v for k, v in cfg.items() if not k.startswith("_")}
        valid = {f.name for f in fields(cls)}
        unknown = set(params) - valid
        if unknown:
            raise KeyError(f"Parâmetros desconhecidos no arquivo: {sorted(unknown)}")
        return cls(**params)

    def _bands(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cf = self.channels_first
        if cf is None:
            # Canais costumam ser poucos (<= 16); altura e largura, muitos.
            cf = image.shape[0] <= 16 and image.shape[-1] > 16
        if cf:
            return image[self.nir_band], image[self.red_band]
        return image[..., self.nir_band], image[..., self.red_band]

    def __call__(self, image: np.ndarray) -> dict:
        nir, red = self._bands(image)
        ndvi = compute_ndvi(nir, red)
        veg = vegetation_mask(ndvi, self.th1, self.th2, self.nlm_h,
                              self.nlm_template, self.nlm_search)
        inst = split_crowns(veg, self.min_distance, self.min_area)
        return {
            "ndvi": ndvi,
            "vegetation": veg,
            "instances": inst,
            "n_crowns": int(inst.max()),
        }



# --------------------------------------------------------------------------
# Configuração
# --------------------------------------------------------------------------
def load_config(path: str | Path) -> dict:
    """Lê um arquivo de configuração JSON (ex.: configs/agml_teacher.json)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Utilidades para as próximas etapas
# --------------------------------------------------------------------------
def instances_to_targets(labels: np.ndarray) -> dict:
    """Converte as etiquetas no formato de alvo do Mask R-CNN.

    Retorna (em NumPy; o laço de treino converte para tensores):
        masks:  (N, H, W) uint8, uma máscara binária por copa.
        boxes:  (N, 4) float32, caixas [x_min, y_min, x_max, y_max].
        labels: (N,) int64, todas = 1 (classe "copa").
    """
    ids = [i for i in np.unique(labels) if i > 0]
    h, w = labels.shape
    masks = np.zeros((len(ids), h, w), np.uint8)
    boxes = np.zeros((len(ids), 4), np.float32)
    for k, i in enumerate(ids):
        m = labels == i
        masks[k] = m
        ys, xs = np.nonzero(m)
        boxes[k] = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
    return {"masks": masks, "boxes": boxes,
            "labels": np.ones(len(ids), np.int64)}


def median_crown_area(labels_list) -> float:
    """Mediana da área (em pixels) das copas do professor.

    Usada na regra da área (regra 1 de Harmon et al., 2022). A mediana é
    usada no lugar da média porque copas fundidas pelo watershed inflariam
    a média.
    """
    areas = []
    for labels in labels_list:
        _, counts = np.unique(labels[labels > 0], return_counts=True)
        areas.extend(counts.tolist())
    return float(np.median(areas)) if areas else 0.0
