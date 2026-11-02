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


# --------------------------------------------------------------------------
# Parte 4b: as transformações e a perda
# --------------------------------------------------------------------------
# Fator de escala do artigo: imagens de 80x80 ampliadas para 96x96 (1,2x).
SCALE = 1.2

# As três transformações do RS-SSKD (equações 3 e 4), na ordem do artigo.
PAPER_TRANSFORMS = ("flip", "scale", "flip+scale")


class Transform:
    """Uma transformação T, aplicada à imagem e aos mapas de objetividade.

    Nomes aceitos (podem ser combinados com "+", aplicados da esquerda para
    a direita):
        flip     espelhamento vertical (de cabeça para baixo), como o t_flip
                 do artigo;
        scale    ampliação de 1,2x por interpolação bilinear;
        rot90, rot180, rot270   rotações (anti-horário), usadas no modo de
                 sorteio do D4;
        hflip    espelhamento horizontal (também do D4).
    """

    def __init__(self, name: str):
        self.name = name
        self.steps = name.split("+")
        unknown = set(self.steps) - {"flip", "hflip", "scale", "rot90", "rot180", "rot270"}
        if unknown:
            raise ValueError(f"Transformação desconhecida: {sorted(unknown)}")

    def __repr__(self):
        return f"Transform({self.name!r})"

    def image(self, x: torch.Tensor) -> torch.Tensor:
        """T(x) para um lote de imagens (B, C, H, W)."""
        for s in self.steps:
            x = _apply(s, x, size=None, is_map=False)
        return x

    def maps(self, m: torch.Tensor, size) -> torch.Tensor:
        """T(mapa): leva o mapa da imagem original para o referencial da
        imagem transformada. `size` é o (h, w) do mapa da transformada; a
        escala é feita redimensionando o mapa para esse tamanho, como no
        cam_consistency do artigo."""
        for s in self.steps:
            m = _apply(s, m, size=size, is_map=True)
        if m.shape[-2:] != tuple(size):
            m = torch.nn.functional.interpolate(m, size=tuple(size), mode="bilinear",
                                                align_corners=False)
        return m


def _apply(step: str, t: torch.Tensor, size, is_map: bool) -> torch.Tensor:
    F = torch.nn.functional
    if step == "flip":
        return t.flip(-2)
    if step == "hflip":
        return t.flip(-1)
    if step == "scale":
        if is_map:
            return t                      # o mapa é ajustado no fim (maps)
        return F.interpolate(t, scale_factor=SCALE, mode="bilinear",
                             align_corners=False)
    k = {"rot90": 1, "rot180": 2, "rot270": 3}[step]
    t = torch.rot90(t, k, dims=(-2, -1))
    if is_map and k % 2 == 1:
        # Girar 90° troca altura com largura: a âncora de proporção 0,5
        # (larga) vira de proporção 2 (alta), e vice-versa. Os canais dos
        # mapas seguem a ordem das proporções (0,5; 1; 2), então trocamos o
        # primeiro com o último.
        t = t.flip(1)
    return t


def choose_transforms(mode: str = "artigo", generator=None) -> list[Transform]:
    """As transformações deste passo de treino.

    mode="artigo":  as três do RS-SSKD, sempre (padrão).
    mode="sorteio": uma das 7 do D4 que não são a identidade, sorteada.
    Também aceita uma lista de nomes separados por vírgula, ex. "flip,rot90".
    """
    if mode == "artigo":
        return [Transform(n) for n in PAPER_TRANSFORMS]
    if mode == "sorteio":
        d4 = ["rot90", "rot180", "rot270", "hflip",
              "rot90+hflip", "rot180+hflip", "rot270+hflip"]
        i = int(torch.randint(len(d4), (), generator=generator))
        return [Transform(d4[i])]
    return [Transform(n.strip()) for n in mode.split(",")]


def consistency_loss(model, images: torch.Tensor, original_maps,
                     transforms) -> tuple[torch.Tensor, dict]:
    """Perda de consistência do RS-SSKD (equação 2), somada sobre as
    transformações (equação 4):

        L_ss = soma_T  média_níveis  MSE( T(P(x)), P(T(x)) )

    em que P é o mapa de objetividade da RPN em probabilidade (sigmoide da
    nota). Usamos a probabilidade, e não a nota bruta, para a perda ficar
    entre 0 e 1 e não ser dominada por notas muito grandes; o artigo compara
    as CAMs brutas, mas lá elas eram comparáveis entre si.

    images:        lote (B, C, H, W) da imagem original.
    original_maps: os mapas da original (lista por nível), capturados no
                   passo normal de treino (capture_objectness).
    transforms:    lista de Transform (choose_transforms).

    O gradiente passa pelos dois lados, como no artigo: as duas
    "ramificações" são a mesma rede.
    """
    total = images.new_zeros(())
    parts = {}
    for t in transforms:
        trans_maps = objectness_maps(model, t.image(images))
        per_level = []
        for orig, trans in zip(original_maps, trans_maps):
            target = t.maps(torch.sigmoid(orig), trans.shape[-2:])
            per_level.append(torch.nn.functional.mse_loss(target, torch.sigmoid(trans)))
        loss_t = torch.stack(per_level).mean()
        parts[t.name] = float(loss_t.detach())
        total = total + loss_t
    return total, parts
