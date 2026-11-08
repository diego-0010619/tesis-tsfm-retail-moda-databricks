"""
Descarga oficial del conjunto de datos real de H&M Personalized Fashion Recommendations
hacia la carpeta física local del proyecto (data/hm/) desde el mirror público de Hugging Face Hub.

Archivos descargados:
1. data/hm/articles.csv                (Catálogo maestro real de 105.542 artículos de H&M)
2. data/hm/articles.parquet            (Catálogo maestro en formato columnar Parquet)
3. data/hm/customers.parquet           (Maestro real de 1.371.980 clientes de H&M)
4. data/hm/transactions_train.parquet  (31.788.324 transacciones reales de H&M 2018-2020)
"""

import os
import sys
import time
import urllib.request
from pathlib import Path

HF_BASE_URL = "https://huggingface.co/datasets/rupikaahere/hnm-fashion-recommendations-data/resolve/main"

FILES_TO_DOWNLOAD = [
    ("data/processed/articles.parquet", "articles.parquet"),
    ("data/raw/articles.csv", "articles.csv"),
    ("data/processed/customers.parquet", "customers.parquet"),
    ("data/processed/transactions_train.parquet", "transactions_train.parquet"),
]


def download_file(url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1024 * 1024:
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"[SKIP] {dest_path.name} ya existe ({size_mb:.2f} MB)")
        return

    print(f"[START] Descargando {dest_path.name} desde {url} ...")
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (UPB-Thesis-Data-Downloader)"})
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    with urllib.request.urlopen(req, timeout=120) as response, open(tmp_path, "wb") as out_file:
        total_size = int(response.info().get("Content-Length", 0))
        downloaded = 0
        chunk_size = 8 * 1024 * 1024  # 8 MB chunks
        last_print = 0

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if downloaded - last_print >= 32 * 1024 * 1024 or downloaded == total_size:
                last_print = downloaded
                elapsed = max(time.time() - t0, 0.001)
                speed = (downloaded / (1024 * 1024)) / elapsed
                if total_size > 0:
                    pct = 100.0 * downloaded / total_size
                    print(
                        f"  -> {dest_path.name}: {downloaded / (1024*1024):.1f} / "
                        f"{total_size / (1024*1024):.1f} MB ({pct:.1f}%) a {speed:.1f} MB/s",
                        flush=True,
                    )

    tmp_path.replace(dest_path)
    elapsed = time.time() - t0
    print(f"[DONE] {dest_path.name} guardado en {dest_path} ({dest_path.stat().st_size / (1024*1024):.2f} MB en {elapsed:.1f}s)")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    hm_dir = project_root / "data" / "hm"
    hm_dir.mkdir(parents=True, exist_ok=True)
    print(f"Carpeta física de destino: {hm_dir}")

    for remote_rel, local_name in FILES_TO_DOWNLOAD:
        url = f"{HF_BASE_URL}/{remote_rel}"
        dest = hm_dir / local_name
        download_file(url, dest)

    print("\n=== RESUMEN DE ARCHIVOS REALES DE H&M EN CARPETA LOCAL ===")
    for p in sorted(hm_dir.glob("*")):
        print(f" - {p.name}: {p.stat().st_size / (1024*1024):.2f} MB")


if __name__ == "__main__":
    main()

