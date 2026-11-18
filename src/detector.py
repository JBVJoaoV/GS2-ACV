"""Classe de uso do FireWatch AI.

Uso em outro projeto:

    from firewatch_ai import FireWatchAI
    fw = FireWatchAI()                 # UMA vez, na inicialização
    resultado = fw.detectar("foto.jpg")
    print(fw.gerar_alerta("foto.jpg"))
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np

from . import config

Imagem = Union[str, Path, bytes, np.ndarray, "Image.Image"]  # noqa: F821


def _resolver_pesos(origem: str) -> str:
    """Aceita caminho local ou 'hf:usuario/repo/arquivo.pt' (Hugging Face Hub)."""
    if origem.startswith("hf:"):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as erro:  # pragma: no cover
            raise ImportError(
                "Para baixar pesos do Hugging Face instale: pip install 'firewatch-ai[hf]'"
            ) from erro
        partes = origem[3:].split("/")
        if len(partes) < 3:
            raise ValueError("Use o formato hf:usuario/repositorio/arquivo.pt")
        repo_id, arquivo = "/".join(partes[:2]), "/".join(partes[2:])
        return hf_hub_download(repo_id=repo_id, filename=arquivo)

    caminho = Path(origem).expanduser()
    if not caminho.is_file():
        raise FileNotFoundError(
            f"Pesos não encontrados: {caminho}\n"
            "Se instalou pelo GitHub, confirme que src/data/ foi enviado "
            "(rode scripts/prepare_package.py e faça commit)."
        )
    return str(caminho)


def _preparar_imagem(imagem: Imagem):
    """Converte bytes em array BGR; os demais tipos o Ultralytics já entende."""
    if isinstance(imagem, (bytes, bytearray)):
        import cv2

        array = np.frombuffer(imagem, dtype=np.uint8)
        decodificada = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if decodificada is None:
            raise ValueError("Não foi possível decodificar a imagem enviada.")
        return decodificada
    if isinstance(imagem, Path):
        return str(imagem)
    return imagem


class FireWatchAI:
    """Detector de fogo (YOLO12n treinado no dataset Fire v3, Roboflow)."""

    def __init__(
        self,
        pesos: Optional[str] = None,
        confianca: Optional[float] = None,
        tamanho_imagem: int = config.TAMANHO_IMAGEM_PADRAO,
        dispositivo: Optional[str] = None,
    ) -> None:
        from ultralytics import YOLO

        self.origem_pesos = pesos or config.pesos_configurados()
        self.confianca = confianca if confianca is not None else config.confianca_configurada()
        self.tamanho_imagem = tamanho_imagem
        self.dispositivo = dispositivo or config.dispositivo_configurado()

        # Carrega o modelo uma única vez.
        self.modelo = YOLO(_resolver_pesos(self.origem_pesos))
        self.classes = dict(self.modelo.names)

    # ------------------------------------------------------------------ imagem
    def _prever(self, fonte):
        argumentos = dict(
            source=fonte,
            conf=self.confianca,
            imgsz=self.tamanho_imagem,
            verbose=False,
        )
        if self.dispositivo is not None:
            argumentos["device"] = self.dispositivo
        return self.modelo.predict(**argumentos)[0]

    def _resultado_para_dict(self, resultado) -> dict[str, Any]:
        deteccoes = []
        for caixa in resultado.boxes:
            classe_id = int(caixa.cls[0])
            deteccoes.append(
                {
                    "classe": self.classes.get(classe_id, str(classe_id)),
                    "confianca": round(float(caixa.conf[0]), 4),
                    "caixa_xyxy": [round(float(v), 1) for v in caixa.xyxy[0].tolist()],
                }
            )
        deteccoes.sort(key=lambda d: d["confianca"], reverse=True)
        altura, largura = resultado.orig_shape[:2]
        return {
            "status": "fogo_detectado" if deteccoes else "sem_fogo",
            "total_deteccoes": len(deteccoes),
            "maior_confianca": deteccoes[0]["confianca"] if deteccoes else 0.0,
            "deteccoes": deteccoes,
            "imagem_largura": int(largura),
            "imagem_altura": int(altura),
            "limiar_confianca": self.confianca,
            "aviso": config.AVISO,
        }

    def detectar(self, imagem: Imagem) -> dict[str, Any]:
        """Analisa uma imagem (caminho, URL, bytes, array BGR ou PIL) e devolve um dict."""
        return self._resultado_para_dict(self._prever(_preparar_imagem(imagem)))

    def gerar_alerta(self, imagem: Imagem) -> str:
        """Mesma análise de detectar(), em texto pronto para exibir."""
        r = self.detectar(imagem)
        if r["status"] == "sem_fogo":
            return f"Nenhum foco de fogo detectado (limiar {r['limiar_confianca']:.0%}).\n{r['aviso']}"
        linhas = [
            f"ALERTA: {r['total_deteccoes']} foco(s) de fogo detectado(s) "
            f"(maior confiança {r['maior_confianca']:.0%})."
        ]
        for i, d in enumerate(r["deteccoes"], 1):
            x1, y1, x2, y2 = d["caixa_xyxy"]
            linhas.append(f"  {i}. {d['classe']} {d['confianca']:.0%} em ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")
        linhas.append(r["aviso"])
        return "\n".join(linhas)

    def anotar(self, imagem: Imagem, saida: Optional[Union[str, Path]] = None) -> np.ndarray:
        """Devolve a imagem com as caixas desenhadas (BGR); salva se 'saida' for informada."""
        anotada = self._prever(_preparar_imagem(imagem)).plot()
        if saida is not None:
            import cv2

            cv2.imwrite(str(saida), anotada)
        return anotada

    def anotar_bytes(self, imagem: Imagem, formato: str = ".jpg") -> bytes:
        """Imagem anotada codificada (útil para APIs web)."""
        import cv2

        ok, buffer = cv2.imencode(formato, self.anotar(imagem))
        if not ok:
            raise RuntimeError("Falha ao codificar a imagem anotada.")
        return buffer.tobytes()

    # ------------------------------------------------------------------- vídeo
    def processar_video(
        self,
        entrada: Union[str, Path],
        saida: Optional[Union[str, Path]] = None,
        mostrar: bool = False,
        progresso: Optional[Callable[[int, int], None]] = None,
    ) -> dict[str, Any]:
        """Gera um vídeo anotado e devolve um resumo com os frames que tiveram fogo."""
        import cv2

        entrada = Path(entrada)
        captura = cv2.VideoCapture(str(entrada))
        if not captura.isOpened():
            raise RuntimeError(f"Não foi possível abrir o vídeo: {entrada}")

        fps = captura.get(cv2.CAP_PROP_FPS) or 25.0
        largura = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
        altura = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
        if largura <= 0 or altura <= 0:
            captura.release()
            raise RuntimeError("Não foi possível identificar a resolução do vídeo.")

        saida = Path(saida) if saida else entrada.with_name(f"{entrada.stem}_firewatch.mp4")
        escritor = cv2.VideoWriter(str(saida), cv2.VideoWriter_fourcc(*"mp4v"), fps, (largura, altura))
        if not escritor.isOpened():
            captura.release()
            raise RuntimeError("Não foi possível criar o vídeo de saída.")

        frames, frames_com_fogo, maior = 0, [], 0.0
        try:
            while True:
                ok, frame = captura.read()
                if not ok:
                    break
                resultado = self._prever(frame)
                if len(resultado.boxes):
                    frames_com_fogo.append(frames)
                    maior = max(maior, float(resultado.boxes.conf.max()))
                anotado = resultado.plot()
                if anotado.shape[1] != largura or anotado.shape[0] != altura:
                    anotado = cv2.resize(anotado, (largura, altura))
                escritor.write(anotado)
                frames += 1
                if progresso:
                    progresso(frames, total)
                if mostrar:
                    cv2.imshow("FireWatch AI", anotado)
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        break
        finally:
            captura.release()
            escritor.release()
            if mostrar:
                cv2.destroyAllWindows()

        return {
            "status": "fogo_detectado" if frames_com_fogo else "sem_fogo",
            "video_saida": str(saida),
            "frames_processados": frames,
            "frames_com_fogo": len(frames_com_fogo),
            "primeiro_frame_com_fogo": frames_com_fogo[0] if frames_com_fogo else None,
            "primeiro_segundo_com_fogo": round(frames_com_fogo[0] / fps, 2) if frames_com_fogo else None,
            "maior_confianca": round(maior, 4),
            "fps": round(fps, 2),
            "aviso": config.AVISO,
        }

    def __repr__(self) -> str:
        return f"FireWatchAI(pesos={self.origem_pesos!r}, confianca={self.confianca}, classes={self.classes})"
