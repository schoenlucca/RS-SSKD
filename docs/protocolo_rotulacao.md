# Protocolo de rotulação das copas (Cerrado)

Para Lucca e Felipe rotularem do mesmo jeito: os dois métodos (o aluno RS-SSKD
e o SAM) serão medidos contra esta referência. Os itens marcados **[a definir]**
ficam para a reunião.

## O material

- `rotulagem/imagens/`: recortes de 512×512 da foto original, em **falsa cor**
  (NIR em vermelho, vermelho em verde, verde em azul). Vegetação viva aparece em
  vermelho ou rosa; solo e estrada em azul-claro; sombra em preto.
- `rotulagem/manifesto.json`: de onde veio cada recorte. **Não apague nem
  edite**; ele é necessário para importar os rótulos.
- Os recortes vêm de voos que não entram no treino nem na validação.
- Quantidade e tamanho: **[a definir]** (sugestão: 50 recortes de 512×512).

## As regras

1. **Uma copa = um polígono**, com a classe `copa`. Não use pincel nem máscara:
   a importação só lê polígonos.
2. **O que é copa:** a copa de uma árvore ou arvoreta, vista de cima. Não entram
   capim, chão, sombra entre copas nem estrada. Arbustos: **[a definir]**
   (sugestão: entram se tiverem mais de ~15 pixels de diâmetro na imagem).
3. **Copas claras ou sem folha também são copas.** Copas rosa-claro,
   esbranquiçadas ou com galhos aparentes entram. O professor (TEBAKA) costuma
   perdê-las, e a referência precisa contá-las, senão o erro dele não aparece.
4. **Copas que se tocam:** separe pela linha de sombra ou pela mudança de textura
   ou de cor entre elas. Se não der para decidir, desenhe uma copa só; não
   invente uma divisão.
5. **Copas na borda do recorte:** rotule a parte visível, até a borda.
6. **Contorno:** acompanhe a borda da copa com folga de 1 ou 2 pixels. Não precisa
   seguir cada folha; o critério de acerto é IoU ≥ 0,5.
7. **Na dúvida se é árvore:** não rotule, e anote o nome da imagem numa lista
   de dúvidas para a reunião.

## Concordância entre vocês

Os **10 primeiros recortes** (em ordem alfabética) são rotulados pelos dois, sem
um ver o do outro. Os outros são divididos. A concordância entre vocês é o teto
realista do F1 e vai para o artigo:

```
python3 scripts/compare_annotators.py rotulagem/lucca.json rotulagem/felipe.json --nomes Lucca Felipe
```

Anotem também o tempo gasto, para descrever o esforço no artigo.

## A ferramenta

**CVAT** (cvat.ai, gratuito):
1. Crie uma tarefa com as imagens de `rotulagem/imagens/` e um rótulo `copa`
   do tipo polígono.
2. Rotule com a ferramenta de polígono (tecla N).
3. Exporte em **COCO 1.0**, que gera um arquivo `instances_default.json`.
   Renomeie para `<seu nome>.json`.

**Roboflow:**
1. Crie um projeto do tipo *Instance Segmentation* e suba as imagens.
2. Rotule com polígonos (*Smart Polygon* ajuda, mas confira cada contorno).
3. Gere a versão **sem pré-processamento e sem aumentos** e exporte em
   **COCO Segmentation**.

A importação desfaz a renomeação e o redimensionamento que o Roboflow faz, mas
é melhor evitá-los.

## Depois de rotular

```
python3 scripts/import_labels.py --coco rotulagem/lucca.json --rotulador Lucca
```

Se cada um exportar o seu arquivo, junte os dois antes ou importe um só.
**[a definir]**: quem consolida o arquivo final.
