"""Copia os pesos treinados para dentro do pacote (src/data/).

Rode antes de cada release e faça commit de src/data/:
    python scripts/prepare_package.py
    python scripts/prepare_package.py --pesos runs/firewatch/yolo12n_fire_smoke/weights/best.pt
"""

import argparse
import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "src" / "data"
NOME = "firewatch_yolo12n_best.pt"
LIMITE_GITHUB = 100 * 1024 * 1024


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pesos", default=str(RAIZ / "Model" / NOME), help="Pesos treinados (best.pt)")
    p.add_argument("--conf", type=float, default=0.25, help="Limiar de confiança padrão")
    a = p.parse_args()

    origem = Path(a.pesos)
    if not origem.is_file():
        raise SystemExit(f"Pesos não encontrados: {origem}")
    tamanho = origem.stat().st_size
    if tamanho >= LIMITE_GITHUB:
        raise SystemExit("Pesos acima de 100 MB: publique no Hugging Face e use FIREWATCH_WEIGHTS=hf:...")

    DESTINO.mkdir(parents=True, exist_ok=True)
    destino = DESTINO / NOME
    shutil.copy2(origem, destino)

    sha256 = hashlib.sha256(destino.read_bytes()).hexdigest()
    metadados = {
        "modelo": "YOLO12n (Ultralytics), fine-tuning",
        "dataset": "Fire v3 (Roboflow vufindr/fire-ztoay, licença MIT)",
        "classes": ["Fire"],
        "imgsz": 640,
        "confianca": a.conf,
        "pesos": NOME,
        "sha256": sha256,
        "tamanho_bytes": tamanho,
        "gerado_em": date.today().isoformat(),
    }
    (DESTINO / "metadata.json").write_text(json.dumps(metadados, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: {destino} ({tamanho / 1e6:.1f} MB, sha256 {sha256[:12]}...)")
    print("Agora: git add src/data && git commit")


if __name__ == "__main__":
    main()
