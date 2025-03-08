# AR Virtual Assistant with Integrated Pac-Man Game

## Overview
Este projeto é um assistente virtual de realidade aumentada (AR) que utiliza técnicas de visão computacional com OpenCV e MediaPipe para detectar gestos das mãos e marcos faciais, permitindo a interação através de diversos modos. Dentre os modos disponíveis, destaca-se a integração de um jogo de Pac-Man, onde o personagem é controlado por gestos da mão em tempo real.

## Funcionalidades

- **Detecção de Mãos**: Utiliza MediaPipe para identificar e rastrear os pontos das mãos, possibilitando interações intuitivas.
- **Detecção de Face**: Emprega o MediaPipe Face Mesh para aplicar filtros e reconhecer gestos faciais.
- **Menu Interativo**: Interface virtual com botões para seleção dos modos: Desenho, Filtros e Jogo.
- **Modo de Desenho**: Permite ao usuário desenhar na tela com gestos, escolhendo cores e limpando a área de desenho.
- **Modo de Filtros**: Aplica diversos filtros faciais (neon, cartoon e máscara) em tempo real.
- **Modo de Jogo (Pac-Man)**: Incorpora um jogo de Pac-Man com labirinto, fantasmas e pontuação, onde o movimento do personagem é controlado pela posição da mão.

## Instalação

### Pré-requisitos
- Python 3.x
- pip (gerenciador de pacotes Python)

### Dependências
Instale as bibliotecas necessárias executando o seguinte comando:

```bash
pip install opencv-python numpy mediapipe pyyaml pygame
```

ou pelo requirements.txt:

```bash
pip install -r requirements.txt
```

## Assets e Configurações

### Assets Visuais:
Coloque as imagens necessárias na pasta `assets/`:
- hat.png
- glasses.png
- mask.png

### Arquivos de Som (Opcional):
Para habilitar os efeitos sonoros do jogo Pac-Man, adicione os seguintes arquivos na raiz do projeto:
- pacman_beginning.wav
- pacman_chomp.wav
- pacman_death.wav

### Arquivo de Configuração:
O projeto utiliza um arquivo `config.yaml` para ajustar parâmetros como fonte do vídeo e níveis de confiança para a detecção. Caso o arquivo não seja encontrado, serão utilizadas configurações padrão.

## Uso

Para executar o projeto, rode o script principal:

```bash
python <nome_do_arquivo>.py
```

### Instruções de Interação

- **Menu Principal**: Ao iniciar o aplicativo, o menu exibirá botões para os modos "Desenhar", "Filtros" e "Jogo". Use a mão para apontar e, realizando um gesto de pinça (aproximando o polegar do indicador), selecione o modo desejado.

- **Modo Desenho**:
  - Selecione uma cor da paleta apresentada na tela.
  - Mantenha o dedo indicador estendido (com os outros dedos fechados) para desenhar.
  - Utilize o botão "Limpar" para apagar os desenhos.

- **Modo Filtros**:
  - Navegue entre os filtros disponíveis usando os botões "< Prev" e "Next >".
  - O filtro selecionado será aplicado à sua face em tempo real.

- **Modo Jogo (Pac-Man)**:
  - No modo de jogo, o Pac-Man é controlado pela posição da ponta do dedo indicador.
  - O labirinto, fantasmas e pontuação são exibidos sobre a imagem da câmera.
  - Pressione a tecla `r` para reiniciar o jogo a qualquer momento.

- **Retorno ao Menu**: Em qualquer modo, toque no botão "Menu" exibido no canto superior esquerdo para retornar à tela principal.

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests para melhorias, correções ou novas funcionalidades.

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).