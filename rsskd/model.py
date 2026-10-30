"""Backbone com Class Activation Maps (CAM)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class RSSSKDNet(nn.Module):
    """ResNet + GAP + FC, devolvendo logits, embedding e CAMs.

    As duas "ramificacoes" do artigo compartilham pesos, entao basta uma
    instancia desta rede aplicada duas vezes (original e transformada).
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 5,
        backbone: str = "resnet101",
        pretrained: bool = True,
    ):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        net = getattr(torchvision.models, backbone)(weights=weights)

        if in_channels != 3:
            net.conv1 = self._adapt_first_conv(net.conv1, in_channels, pretrained)

        self.features = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4,
        )
        self.embed_dim = net.fc.in_features  # 2048 na ResNet-101
        self.fc = nn.Linear(self.embed_dim, num_classes)

    @staticmethod
    def _adapt_first_conv(conv: nn.Conv2d, in_channels: int, pretrained: bool) -> nn.Conv2d:
        """Troca a 1a convolucao para aceitar bandas extras (NIR, Red-Edge).

        Com pesos pre-treinados, os canais RGB sao mantidos e cada canal extra
        comeca com a media dos filtros RGB, para nao perder o pre-treino.
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
                n = min(3, in_channels)
                new.weight[:, :n] = conv.weight[:, :n]
                if in_channels > 3:
                    mean = conv.weight.mean(dim=1, keepdim=True)
                    new.weight[:, 3:] = mean.repeat(1, in_channels - 3, 1, 1)
        return new

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embedding usado na etapa few-shot (saida do GAP)."""
        return self.features(x).mean(dim=(2, 3))

    def forward(self, x: torch.Tensor, return_cam: bool = False):
        fmap = self.features(x)                 # B x C x H x W
        emb = fmap.mean(dim=(2, 3))             # GAP -> B x C
        logits = self.fc(emb)                   # B x L
        if not return_cam:
            return logits, emb
        # Eq. (1) do artigo: M_c(x, y) = sum_k w_k^c f_k(x, y).
        # Equivale a aplicar a FC como convolucao 1x1 sobre o mapa de features.
        cam = F.conv2d(fmap, self.fc.weight[:, :, None, None])  # B x L x H x W
        return logits, emb, cam
