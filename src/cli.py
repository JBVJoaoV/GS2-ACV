"""Linha de comando: `firewatch foto.jpg` ou `firewatch video.mp4 --saida out.mp4`."""

import argparse
import json
import sys
from pathlib import Path

EXT_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="firewatch", description="Detecção de fogo em imagem ou vídeo.")
    p.add_argument("arquivo", help="Imagem ou vídeo a analisar")
    p.add_argument("--saida", help="Arquivo anotado de saída")
    p.add_argument("--conf", type=float, help="Limiar de confiança (padrão 0.25)")
    p.add_argument("--pesos", help="Caminho dos pesos ou hf:usuario/repo/arquivo.pt")
    p.add_argument("--json", action="store_true", help="Imprime o resultado em JSON")
    p.add_argument("--mostrar", action="store_true", help="Mostra o vídeo durante o processamento")
    a = p.parse_args(argv)

    from .detector import FireWatchAI

    fw = FireWatchAI(pesos=a.pesos, confianca=a.conf)
    caminho = Path(a.arquivo)

    if caminho.suffix.lower() in EXT_VIDEO:
        def progresso(n, total):
            if n % 30 == 0:
                print(f"Frames: {n}/{total or '?'}", file=sys.stderr)

        r = fw.processar_video(caminho, a.saida, mostrar=a.mostrar, progresso=progresso)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return 0

    if a.saida:
        fw.anotar(caminho, a.saida)
    if a.json:
        print(json.dumps(fw.detectar(caminho), ensure_ascii=False, indent=2))
    else:
        print(fw.gerar_alerta(caminho))
    return 0


if __name__ == "__main__":
    sys.exit(main())
