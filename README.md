# RS-SSKD no Cerrado

Implementação do RS-SSKD para classificação few-shot de espécies arbóreas do
Cerrado em imagens de drone (PIBIC, Plano de Trabalho 1.2).

Método de referência: Zhang, Li, Wang e Wang, "RS-SSKD: Self-Supervision
Equipped with Knowledge Distillation for Few-Shot Remote Sensing Scene
Classification", Sensors 21(5), 1566, 2021.
https://www.mdpi.com/1424-8220/21/5/1566

## Instalação

```
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Para treinar com GPU, instale o PyTorch com CUDA seguindo https://pytorch.org.

## Teste

```
pytest
```

## Estrutura

| Caminho | Conteúdo |
| --- | --- |
| `rsskd/model.py` | `RSSSKDNet`: ResNet com GAP, FC e Class Activation Maps |
| `rsskd/losses.py` | Perda de consistência de CAM e perda de auto-destilação |
| `rsskd/fewshot.py` | Protótipos com cosseno e amostrador de episódios |
| `tests/test_smoke.py` | Teste com dados sintéticos |

## Situação

Feito: blocos do método (modelo, perdas, classificador few-shot, episódios).

A fazer: carregamento de dados, laços de treino (base, destilação,
meta-aprendizagem), reprodução no NWPU-RESISC45 e experimentos no Cerrado.

## Diferenças em relação ao artigo

- Backbone ResNet-101 (o artigo usa ResNet-12), configurável.
- Entrada com 5 canais (RGB, NIR e Red-Edge); o artigo usa RGB.
