"""Testes com o modelo real (os pesos estão no pacote; roda em CPU em segundos)."""

from pathlib import Path

import numpy as np
import pytest

from firewatch_ai import FireWatchAI

RAIZ = Path(__file__).resolve().parents[1]
IMAGENS_TESTE = sorted((RAIZ / "Fire.v3i.yolov11" / "test" / "images").glob("*.jpg"))


@pytest.fixture(scope="module")
def fw():
    return FireWatchAI(dispositivo="cpu")


def test_classes(fw):
    assert list(fw.classes.values()) == ["Fire"]


def test_imagem_preta_sem_fogo(fw):
    r = fw.detectar(np.zeros((640, 640, 3), dtype=np.uint8))
    assert r["status"] == "sem_fogo"
    assert r["deteccoes"] == []
    assert "193" in r["aviso"]


@pytest.mark.skipif(not IMAGENS_TESTE, reason="dataset de teste ausente")
def test_detecta_fogo_no_conjunto_de_teste(fw):
    positivos = sum(fw.detectar(p)["status"] == "fogo_detectado" for p in IMAGENS_TESTE[:10])
    assert positivos >= 7


@pytest.mark.skipif(not IMAGENS_TESTE, reason="dataset de teste ausente")
def test_bytes_e_alerta(fw):
    dados = IMAGENS_TESTE[0].read_bytes()
    r = fw.detectar(dados)
    assert r["status"] in {"fogo_detectado", "sem_fogo"}
    assert isinstance(fw.gerar_alerta(dados), str)
    assert fw.anotar_bytes(dados)[:2] == b"\xff\xd8"  # JPEG


def test_bytes_invalidos(fw):
    with pytest.raises(ValueError):
        fw.detectar(b"nao e imagem")


def test_pesos_inexistentes():
    with pytest.raises(FileNotFoundError):
        FireWatchAI(pesos="nao_existe.pt")
