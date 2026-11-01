"""O aluno: Mask R-CNN com ResNet-101 + FPN e entrada de 6 canais.

É a primeira peça do item 3 (o modelo da etapa 1 do aluno).

Como o Mask R-CNN funciona, em quatro blocos:

    imagem (6, 256, 256)
      │
      ▼
    1. backbone: ResNet-101 + FPN
       A ResNet-101 extrai mapas de características em 4 resoluções
       (1/4, 1/8, 1/16 e 1/32 da imagem). A FPN (Feature Pyramid Network)
       combina essas resoluções, para que copas pequenas sejam vistas nos
       mapas finos e copas grandes nos mapas grossos.
      │
      ▼
    2. RPN (Region Proposal Network)
       Em cada posição dos mapas, testa "âncoras" (caixas de tamanhos e
       proporções fixas) e propõe as regiões que parecem conter um objeto.
      │
      ▼
    3. cabeça de caixas
       Para cada proposta, recorta as características (RoIAlign), decide se
       é copa ou fundo e ajusta a caixa.
      │
      ▼
    4. cabeça de máscaras
       Para cada caixa aceita, prevê uma máscara de 28x28 dentro da caixa,
       depois ampliada para o tamanho da caixa na imagem. É isso que dá o
       contorno de cada copa, e não só a caixa.

Três adaptações ao nosso problema, feitas aqui:

  - **6 canais na entrada.** A ResNet-101 pré-treinada no ImageNet espera
    RGB. A primeira convolução é trocada por uma de 6 canais: os 3 primeiros
    copiam os filtros RGB pré-treinados, e os novos (NIR, Red-Edge, NDVI)
    começam com a média dos filtros RGB. Tudo é multiplicado por 3/6, para
    a saída da camada ficar na mesma escala do pré-treino.
  - **Sem normalização interna.** O Mask R-CNN do torchvision normaliza a
    imagem com as médias do ImageNet. Nós já normalizamos (parte 2a), então
    a média vira 0 e o desvio 1, e a imagem passa intacta.
  - **Âncoras do tamanho das copas.** As âncoras padrão (32 a 512 pixels)
    foram pensadas para fotos comuns; 512 é maior que a nossa imagem
    inteira. As copas do professor têm de ~22 a ~55 pixels de lado (raiz
    das áreas p10 e p90), então usamos 16, 32, 64, 128 e 256.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet101_Weights, resnet101
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.ops import FrozenBatchNorm2d

# Um tamanho de âncora por nível da FPN (do mais fino ao mais grosso).
ANCHOR_SIZES = ((16,), (32,), (64,), (128,), (256,))
ASPECT_RATIOS = ((0.5, 1.0, 2.0),) * len(ANCHOR_SIZES)


def adapt_first_conv(conv: nn.Conv2d, in_channels: int, pretrained: bool) -> nn.Conv2d:
    """Troca a 1a convolução da ResNet por uma com `in_channels` canais.

    Com pesos pré-treinados: os canais 0..2 recebem os filtros RGB, os
    canais extras recebem a média deles, e tudo é escalado por 3/in_channels.
    Sem a escala, a soma sobre 6 canais daria uma saída ~2x maior que a do
    pré-treino, e a BatchNorm congelada logo depois (com as estatísticas do
    ImageNet) ficaria descalibrada.
    """
    new = nn.Conv2d(
        in_channels, conv.out_channels,
        kernel_size=conv.kernel_size,  # type: ignore[arg-type]
        stride=conv.stride,  # type: ignore[arg-type]
        padding=conv.padding,  # type: ignore[arg-type]
        bias=False,
    )
    if pretrained:
        with torch.no_grad():
            w = conv.weight                                  # (64, 3, 7, 7)
            n = min(3, in_channels)
            new.weight[:, :n] = w[:, :n]
            if in_channels > 3:
                new.weight[:, 3:] = w.mean(dim=1, keepdim=True)
            new.weight.mul_(3.0 / in_channels)
    return new


def build_student(in_channels: int = 6, pretrained_backbone: bool = True,
                  trainable_layers: int = 3, image_size: int = 256,
                  max_detections: int = 100) -> MaskRCNN:
    """Monta o Mask R-CNN do aluno.

    in_channels:         canais da entrada (6: 5 bandas + NDVI).
    pretrained_backbone: ResNet-101 com pesos do ImageNet (baixa ~170 MB na
                         primeira vez). False nos testes.
    trainable_layers:    quantos dos 5 estágios da ResNet treinam (contando
                         do fim). 3 = layer2, layer3 e layer4 treinam; os
                         estágios iniciais, que detectam bordas e texturas
                         genéricas, ficam congelados. A 1a convolução treina
                         sempre, porque os filtros dos canais novos não
                         vieram do pré-treino.
    image_size:          lado da imagem. O Mask R-CNN redimensiona as
                         imagens para este tamanho; igual ao das nossas
                         imagens (256), ele não redimensiona nada.
    max_detections:      máximo de copas por imagem na previsão.

    Classes: 2 (0 = fundo, 1 = copa). Não há classificação de espécies.
    """
    weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    # Com pré-treino, a BatchNorm fica congelada (estatísticas do ImageNet):
    # com lotes de 2 a 4 imagens, estimar as estatísticas do lote seria ruim.
    norm = FrozenBatchNorm2d if pretrained_backbone else nn.BatchNorm2d
    body = resnet101(weights=weights, norm_layer=norm)
    body.conv1 = adapt_first_conv(body.conv1, in_channels, pretrained_backbone)

    backbone = _resnet_fpn_extractor(body, trainable_layers)
    # O extrator congela os estágios iniciais, inclusive a 1a convolução;
    # ela volta a treinar (é o mesmo objeto que está dentro do backbone).
    body.conv1.weight.requires_grad_(True)

    return MaskRCNN(
        backbone,
        num_classes=2,
        rpn_anchor_generator=AnchorGenerator(ANCHOR_SIZES, ASPECT_RATIOS),
        image_mean=[0.0] * in_channels,
        image_std=[1.0] * in_channels,
        min_size=image_size,
        max_size=image_size,
        box_detections_per_img=max_detections,
    )


def count_parameters(model: nn.Module) -> dict:
    """Parâmetros totais e treináveis (para o relatório)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "treinaveis": trainable}
