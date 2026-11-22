# Instruções para agentes neste repositório

## Sobre o projeto
Implementações de LSGAN (Mao et al., 2017) para o banco SVHN, escritas originalmente em 2017 com TensorFlow 1.x.

## Regras
- Escreva comentários de Issue e PR, mensagens de commit e descrições de PR em **português do Brasil**.
- Nunca faça commit direto no `master`. Trabalhe sempre num branch e entregue a mudança como Pull Request.
- O framework de destino é **PyTorch**. Os arquivos originais em TensorFlow 1.x ficam preservados na pasta `legacy/`, sem alterações.
- Antes de abrir o PR, rode o código para conferir que funciona. Como o runner não tem GPU, um teste curto basta (por exemplo, poucos passos de treino com dados sintéticos).
- Não coloque dados grandes (como os arquivos `.mat` do SVHN) nem modelos treinados no repositório.
- Na descrição do PR, explique o que mudou e como foi testado.
- Quando o PR resolver uma Issue, escreva `Closes #N` na descrição, **em inglês**, para que o GitHub feche a Issue no merge. Palavras em português (como "Fecha #N") não funcionam.