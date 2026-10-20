"""Unduh Imagenette 160 px: python -m utils.download_data."""

import argparse
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from utils.paths import resolve_data_dir, validate_dataset

# Tautan Imagenette 160 px yang tercantum di https://github.com/fastai/imagenette.
DATASET_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"


def download_dataset(data_dir=None):
    """Gunakan data lengkap yang ada, atau unduh dan validasi di folder sementara."""
    data_dir = resolve_data_dir(data_dir)
    if data_dir.exists():
        try:
            counts = validate_dataset(data_dir)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(
                f"Folder tujuan sudah ada tetapi belum lengkap: {data_dir}. "
                "Pindahkan folder tersebut atau pilih --data-dir yang belum ada, "
                "kemudian ulangi pengunduhan. Data lama tidak ditimpa."
            ) from exc
        print(f"Dataset sudah lengkap: {data_dir} (train={counts['train']}, val={counts['val']}).")
        return data_dir

    data_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="imagenette-download-", dir=data_dir.parent) as temp:
        temp_dir = Path(temp)
        archive = temp_dir / "imagenette2-160.tgz"
        request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "CBAM-Imagenette/1.0"})
        print(f"Mengunduh {DATASET_URL} ...", flush=True)
        with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        print("Mengekstrak dan memeriksa dataset ...", flush=True)
        with tarfile.open(archive, "r:gz") as source:
            source.extractall(path=temp_dir, filter="data")
        extracted = temp_dir / "imagenette2-160"
        validate_dataset(extracted)
        extracted.rename(data_dir)
    print(f"Dataset siap: {data_dir}")
    return data_dir


def main():
    parser = argparse.ArgumentParser(description="Unduh dataset Imagenette versi 160 px.")
    parser.add_argument("--data-dir", type=Path, help="Folder tujuan dataset (default: data/imagenette2-160 di proyek).")
    args = parser.parse_args()
    try:
        download_dataset(args.data_dir)
    except (OSError, ValueError, tarfile.TarError) as exc:
        parser.exit(1, f"Pengunduhan gagal: {exc}\n")


if __name__ == "__main__":
    main()
