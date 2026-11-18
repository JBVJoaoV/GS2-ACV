"""Serviço HTTP para plataformas que não são Python.

Subir:  uvicorn firewatch_ai.api:app --host 0.0.0.0 --port 8000
        (requer: pip install "firewatch-ai[api] @ git+https://github.com/JBVJoaoV/GS2-ACV.git")

Se a variável FIREWATCH_API_TOKEN estiver definida, as rotas exigem o cabeçalho
`Authorization: Bearer <token>`. Exposto na rede: use também HTTPS (proxy reverso).
"""

import os
import secrets
from typing import Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import Response

from .detector import FireWatchAI

TAMANHO_MAXIMO = 10 * 1024 * 1024  # 10 MB por imagem

app = FastAPI(title="FireWatch AI", version="0.1.0")
_detector: Optional[FireWatchAI] = None


def detector() -> FireWatchAI:
    global _detector
    if _detector is None:  # carrega uma vez, na primeira chamada
        _detector = FireWatchAI()
    return _detector


def autenticar(authorization: Optional[str] = Header(default=None)) -> None:
    token = os.environ.get("FIREWATCH_API_TOKEN")
    if not token:
        return
    recebido = (authorization or "").removeprefix("Bearer ").strip()
    if not secrets.compare_digest(recebido, token):
        raise HTTPException(status_code=401, detail="Token inválido ou ausente.")


async def _ler(arquivo: UploadFile) -> bytes:
    conteudo = await arquivo.read()
    if not conteudo:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")
    if len(conteudo) > TAMANHO_MAXIMO:
        raise HTTPException(status_code=413, detail="Imagem maior que 10 MB.")
    return conteudo


@app.on_event("startup")
def _carregar() -> None:
    detector()


@app.get("/saude")
def saude():
    d = detector()
    return {"status": "ok", "classes": d.classes, "limiar_confianca": d.confianca}


@app.post("/detectar", dependencies=[Depends(autenticar)])
async def detectar(imagem: UploadFile = File(...)):
    try:
        return detector().detectar(await _ler(imagem))
    except ValueError as erro:
        raise HTTPException(status_code=400, detail=str(erro))


@app.post("/detectar/imagem", dependencies=[Depends(autenticar)])
async def detectar_imagem(imagem: UploadFile = File(...), formato: str = Query(".jpg", pattern=r"^\.(jpg|png)$")):
    try:
        dados = detector().anotar_bytes(await _ler(imagem), formato)
    except ValueError as erro:
        raise HTTPException(status_code=400, detail=str(erro))
    return Response(content=dados, media_type="image/jpeg" if formato == ".jpg" else "image/png")
