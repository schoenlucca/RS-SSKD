"""Consistência a transformações do RS-SSKD, adaptada ao Mask R-CNN (item 4).

No RS-SSKD original (Zhang et al., 2021, equação 2), a perda compara CAMs:
o mapa de ativação da classe na imagem transformada deve ser a
transformação do mapa da imagem original,

    L_ss = || T(CAM(x)) - CAM(T(x)) ||^2 ,

com T = espelhamento, escala de 1,2x e os dois juntos (equações 3 e 4).

**O equivalente da CAM no Mask R-CNN: o mapa de objetividade da RPN.** A
RPN (Region Proposal Network) aplica, em cada posição de cada nível da FPN,
um pequeno classificador que responde "há uma copa centrada aqui?", uma
nota por âncora. O resultado é um mapa denso de uma classe (copa contra
fundo), calculado por um classificador em cada posição: o mesmo papel da
CAM. Formato, para uma imagem 256x256 (níveis da FPN com passo 4 a 64):

    nível '0': (B, 3, 64, 64)    copas pequenas
    nível '1': (B, 3, 32, 32)
    nível '2': (B, 3, 16, 16)
    nível '3': (B, 3, 8, 8)
    nível 'pool': (B, 3, 4, 4)   copas grandes

Os 3 canais são as 3 proporções de âncora (altura/largura = 0,5, 1 e 2).

Esta primeira parte (4a) só extrai esses mapas:
  - `objectness_maps` roda o backbone e a cabeça da RPN numa imagem, sem as
    cabeças de caixa e de máscara. É o que se usa nas imagens transformadas:
    mais barato que o modelo inteiro.
  - `capture_objectness` "escuta" a RPN durante o passo normal de treino e
    guarda os mapas da imagem original, sem rodar a rede de novo.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch


def objectness_maps(model, x: torch.Tensor) -> list[torch.Tensor]:
    """Mapas de objetividade da RPN para um lote x (B, C, H, W).

    Usa a imagem como está: sem a normalização nem o redimensionamento
    internos do Mask R-CNN. Isso importa para a transformação de escala:
    uma imagem ampliada para 307x307 continua com 307x307, e os mapas saem
    maiores na mesma proporção. (Nossa normalização já foi feita antes, na
    entrada, e a interna do modelo está desligada.)

    Devolve uma lista com um tensor (B, A, h, w) por nível da FPN, com
    gradiente: a perda de consistência treina o backbone e a RPN.
    """
    features = model.backbone(x)
    objectness, _ = model.rpn.head(list(features.values()))
    return objectness


@contextmanager
def capture_objectness(model):
    """Guarda os mapas de objetividade que a RPN calcula durante um forward.

    Uso:
        with capture_objectness(model) as cap:
            perdas = model(imagens, alvos)       # passo normal de treino
        mapas = cap["objectness"]                # da imagem original

    Funciona com um "gancho" (forward hook) na cabeça da RPN: toda vez que
    ela roda, a saída é copiada para `cap`. Os mapas continuam ligados ao
    grafo de gradientes, então podem entrar numa perda.
    """
    cap: dict = {}

    def hook(_module, _inputs, output):
        cap["objectness"] = output[0]

    handle = model.rpn.head.register_forward_hook(hook)
    try:
        yield cap
    finally:
        handle.remove()
