# FireWatch AI (GS2-ACV)

Detecção de fogo em imagens e vídeos com **YOLO12n** treinado (fine-tuning) no dataset *Fire v3* (Roboflow, licença MIT). Uma classe: `Fire`.

O projeto é distribuído como **pacote pip**: outro projeto instala com um comando e usa uma classe, sem copiar notebook nem pesos à mão.

## Instalação

**Rápida (recomendada):** wheel publicado na Release do GitHub. Baixa só ~5 MB de pacote:

```bash
pip install https://github.com/JBVJoaoV/GS2-ACV/releases/download/v0.1.0/firewatch_ai-0.1.0-py3-none-any.whl
# com a API HTTP:
pip install "firewatch-ai[api] @ https://github.com/JBVJoaoV/GS2-ACV/releases/download/v0.1.0/firewatch_ai-0.1.0-py3-none-any.whl"
```

**Pelo repositório (lenta):** clona o repositório inteiro, com dataset e vídeos (~100 MB):

```bash
pip install "git+https://github.com/JBVJoaoV/GS2-ACV.git"
```

**Máquina sem GPU (Windows/Linux):** instale antes o PyTorch de CPU. Ele é bem menor que a versão com CUDA, que o pip baixaria por padrão no Linux:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install https://github.com/JBVJoaoV/GS2-ACV/releases/download/v0.1.0/firewatch_ai-0.1.0-py3-none-any.whl
```

No Colab, o PyTorch já vem instalado: só o `ultralytics` e o pacote são baixados.

Os pesos (`firewatch_yolo12n_best.pt`, 5,5 MB) vêm **dentro do pacote** (`src/data/`). Não há download na primeira execução e funciona offline.

## Uso em Python

```python
from firewatch_ai import FireWatchAI

fw = FireWatchAI()                       # carregue UMA vez, na inicialização

r = fw.detectar("foto.jpg")              # caminho, URL, bytes, array BGR (OpenCV) ou PIL
r["status"]            # "fogo_detectado" | "sem_fogo"
r["maior_confianca"]   # 0.0 a 1.0
r["deteccoes"]         # [{"classe": "Fire", "confianca": 0.79, "caixa_xyxy": [x1, y1, x2, y2]}, ...]

print(fw.gerar_alerta("foto.jpg"))       # texto pronto, com aviso
fw.anotar("foto.jpg", "foto_anotada.jpg")  # imagem com as caixas desenhadas
resumo = fw.processar_video("video.mp4") # gera video_firewatch.mp4 e devolve um resumo
```

Parâmetros: `FireWatchAI(pesos=None, confianca=0.25, tamanho_imagem=640, dispositivo=None)`. Em `dispositivo`, use `"cpu"`, `"0"` (primeira GPU) ou deixe `None` para escolha automática.

## Linha de comando

```bash
firewatch foto.jpg
firewatch foto.jpg --json --saida anotada.jpg
firewatch video.mp4 --saida video_anotado.mp4 --conf 0.3
```

## Serviço HTTP (plataformas que não são Python)

```bash
export FIREWATCH_API_TOKEN="um-token-forte"     # opcional, mas obrigatório se exposto na rede
uvicorn firewatch_ai.api:app --host 0.0.0.0 --port 8000
```

| Rota | Entrada | Saída |
|---|---|---|
| `GET /saude` | — | classes e limiar |
| `POST /detectar` | `multipart/form-data`, campo `imagem` | JSON igual ao de `detectar()` |
| `POST /detectar/imagem` | idem (`?formato=.png` opcional) | imagem anotada |

```bash
curl -H "Authorization: Bearer $FIREWATCH_API_TOKEN" -F "imagem=@foto.jpg" http://localhost:8000/detectar
```

Documentação interativa: `http://localhost:8000/docs`. Se expor fora da máquina local, use HTTPS (proxy reverso) e o token.

## Variáveis de ambiente

| Variável | Efeito |
|---|---|
| `FIREWATCH_WEIGHTS` | outros pesos: caminho local ou `hf:usuario/repo/arquivo.pt` (requer `[hf]`) |
| `FIREWATCH_CONF` | limiar de confiança padrão |
| `FIREWATCH_DEVICE` | `cpu` ou índice da GPU |
| `FIREWATCH_API_TOKEN` | exige `Authorization: Bearer` na API |

## Para quem mantém o projeto

Depois de treinar novos pesos:

```bash
python scripts/prepare_package.py --pesos caminho/para/best.pt   # copia para src/data/ e gera metadata.json
pip install -e ".[dev,api]"
pytest -q
git add src pyproject.toml scripts tests README.md .gitignore
git commit -m "Atualiza pesos do FireWatch"
git push

# gerar o wheel para a instalação rápida
pip install build
python -m build --wheel        # cria dist/firewatch_ai-0.1.0-py3-none-any.whl
```

Publique o wheel: no GitHub, abra **Releases → Draft a new release**, crie a tag `v0.1.0` e anexe o arquivo `dist/firewatch_ai-0.1.0-py3-none-any.whl`. Ao mudar a versão em `pyproject.toml`, ajuste a tag e o nome do arquivo nos comandos de instalação.

### Teste do zero (Colab ou outra máquina)

```python
!pip install -q "git+https://github.com/JBVJoaoV/GS2-ACV.git"
from firewatch_ai import FireWatchAI
import numpy as np
fw = FireWatchAI()
print(fw.detectar(np.zeros((640, 640, 3), dtype=np.uint8))["status"])   # deve dar "sem_fogo"
print(fw.gerar_alerta("sua_foto_com_fogo.jpg"))                          # envie uma foto com fogo: deve dar ALERTA
```

Se falhar, as causas comuns são: `src/data/` não enviado ao GitHub (erro "Pesos não encontrados") ou repositório privado.

## Limitações e cuidados

- O modelo detecta apenas a classe `Fire` (não detecta fumaça). Há falsos positivos (luzes, pôr do sol, objetos laranja) e falsos negativos (fogo pequeno ou encoberto).
- O resultado é **apoio à decisão** e não substitui confirmação humana. Em emergência: **193** (Bombeiros) ou **199** (Defesa Civil).
- Ultralytics/YOLO é distribuído sob AGPL-3.0. Para uso comercial fechado, confira a licença.
- Imagens de câmeras podem conter pessoas: a API não guarda as imagens nem as registra em log. Mantenha assim por causa da LGPD.
