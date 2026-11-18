"""Configurações padrão do FireWatch AI.

Tudo pode ser sobrescrito por argumento da classe ou por variável de ambiente.
"""

import json
import os
from pathlib import Path

PACOTE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACOTE_DIR / "data"

# Pesos embutidos no pacote (5,5 MB: cabem no GitHub, não precisam do Hugging Face).
PESOS_NOME = "firewatch_yolo12n_best.pt"
PESOS_PADRAO = DATA_DIR / PESOS_NOME

# Valores usados no treinamento/teste (02_Teste_Modelo.ipynb).
CONFIANCA_PADRAO = 0.25
TAMANHO_IMAGEM_PADRAO = 640

AVISO = (
    "Detecção automática de apoio. Pode haver falsos positivos e negativos; "
    "confirme visualmente. Emergência: Corpo de Bombeiros 193 / Defesa Civil 199."
)


def _ler_metadados() -> dict:
    arquivo = DATA_DIR / "metadata.json"
    if arquivo.is_file():
        try:
            return json.loads(arquivo.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
    return {}


METADADOS = _ler_metadados()


def pesos_configurados() -> str:
    """Origem dos pesos: argumento > FIREWATCH_WEIGHTS > pesos embutidos."""
    return os.environ.get("FIREWATCH_WEIGHTS", str(PESOS_PADRAO))


def confianca_configurada() -> float:
    valor = os.environ.get("FIREWATCH_CONF")
    if valor is not None:
        return float(valor)
    return float(METADADOS.get("confianca", CONFIANCA_PADRAO))


def dispositivo_configurado():
    """'cpu', '0' (primeira GPU) ou None para o Ultralytics escolher sozinho."""
    return os.environ.get("FIREWATCH_DEVICE") or None
