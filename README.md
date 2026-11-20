# FireWatch AI (GS2-ACV)

Detecção de fogo em imagens e vídeos com visão computacional. O projeto treina um modelo **YOLO12n** para localizar chamas e o entrega como um **pacote Python** (`firewatch-ai`). Qualquer sistema pode instalá-lo com um comando e usá-lo por uma classe, pela linha de comando ou por uma API HTTP.

> **Aviso:** o FireWatch AI é uma ferramenta de apoio. Ele pode errar, tanto com falsos alarmes quanto deixando de ver fogo. Toda detecção deve ser confirmada por uma pessoa. Em emergência: **193** (Corpo de Bombeiros) ou **199** (Defesa Civil).

## Integrantes

| Nome | RM |
|---|---|
| João Vieira | RM 99805 |
| Lucas Speranzini | RM 98297 |

## Sobre o projeto

O modelo parte do YOLO12n pré-treinado e foi ajustado (fine-tuning) com o dataset *Fire v3* do Roboflow: 611 imagens com fogo anotado, divididas em 428 de treino, 122 de validação e 61 de teste. Ele marca cada foco com uma caixa e uma porcentagem de confiança, e processa vídeos quadro a quadro.

Resultado no conjunto de teste (61 imagens, 70 focos anotados):

| Precisão | Recall | mAP50 | mAP50-95 |
|---|---|---|---|
| 0,774 | 0,914 | 0,929 | 0,727 |

**Limitações principais:**
- O modelo detecta só a classe `Fire`, não detecta fumaça.
- O dataset não tem imagens sem fogo, então a taxa de falso alarme em cenas comuns ainda não foi medida.

## Ferramentas

| Ferramenta | Uso no projeto |
|---|---|
| Python 3.10+ | linguagem do projeto |
| Ultralytics YOLO12n | arquitetura de detecção, treino e inferência |
| PyTorch | base de execução do modelo (CPU ou GPU CUDA) |
| OpenCV | leitura e gravação de imagens e vídeos, desenho das caixas |
| Roboflow | origem e anotação do dataset *Fire v3* |
| Google Colab (GPU) e Jupyter / VS Code | treinamento e testes nos notebooks |
| setuptools / pip | empacotamento e distribuição (`pyproject.toml`) |
| FastAPI + Uvicorn | API HTTP opcional |

## Estrutura do repositório

```
GS2-ACV/
├── src/                       # código do pacote (importado como firewatch_ai)
│   ├── detector.py            # classe FireWatchAI: detectar, gerar_alerta, anotar, processar_video
│   ├── config.py              # caminhos, limiar de confiança, variáveis de ambiente
│   ├── cli.py                 # comando `firewatch`
│   ├── api.py                 # API HTTP (FastAPI)
│   └── data/                  # pesos treinados (5,5 MB) + metadata.json
├── notebooks/
│   ├── FireWatch_AI_Treinamento_dataset_local.ipynb   # treinamento e avaliação
│   └── 02_Teste_Modelo.ipynb                          # testes com imagem e vídeo
├── scripts/prepare_package.py # copia novos pesos para dentro do pacote
├── Model/                     # pesos originais do treinamento
├── Fire.v3i.yolov11/          # dataset (train / valid / test)
├── Teste/                     # imagens e vídeos de demonstração
├── docs/                      # relatório e vídeo da solução funcionando
├── output.png                 # curvas de treinamento
└── pyproject.toml             # configuração do pacote
```

## Instalação

Requisitos: Python 3.10 ou mais recente. A GPU é opcional, porque o modelo também roda em CPU.

### Opção A: só usar o detector em outro projeto

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)

# Rápida: wheel da Release (cerca de 5 MB de pacote)
pip install https://github.com/JBVJoaoV/GS2-ACV/releases/download/v0.1.0/firewatch_ai-0.1.0-py3-none-any.whl

# Alternativa: direto do repositório (mais lenta, baixa dataset e vídeos)
pip install "git+https://github.com/JBVJoaoV/GS2-ACV.git"
```

Em máquina sem GPU, rode antes `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`, que é uma versão bem menor do PyTorch.

### Opção B: rodar o projeto completo (código, notebooks e treino)

```bash
git clone https://github.com/JBVJoaoV/GS2-ACV.git
cd GS2-ACV
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[api,dev]"     # instala o pacote em modo edição, com a API e o pytest
pip install jupyter             # para abrir os notebooks
```

Para treinar de novo, abra `notebooks/FireWatch_AI_Treinamento_dataset_local.ipynb`. O recomendado é usar GPU, por exemplo no Colab com T4.

## Uso rápido

```python
from firewatch_ai import FireWatchAI

fw = FireWatchAI()                        # carregar uma vez só
print(fw.gerar_alerta("foto.jpg"))        # texto com os focos e o aviso
resultado = fw.detectar("foto.jpg")       # dict: status, maior_confianca, deteccoes
fw.processar_video("video.mp4")           # gera video_firewatch.mp4 anotado
```

```bash
firewatch foto.jpg                                  # linha de comando
uvicorn firewatch_ai.api:app --port 8000            # API HTTP: POST /detectar
```

## Documentação

- [`docs/DOCUMENTACAO.md`](docs/DOCUMENTACAO.md): referência completa, com métodos, API, variáveis de ambiente, manutenção, limitações e licenças.
- [`docs/Relatorio_FireWatch_AI.pdf`](docs/Relatorio_FireWatch_AI.pdf): relatório do projeto.

## Licenças

O código é distribuído sob AGPL-3.0, a mesma licença do Ultralytics YOLO, de que ele depende. O dataset *Fire v3* (Roboflow) é MIT.