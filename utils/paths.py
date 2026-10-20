"""Lokasi proyek dan pemeriksaan struktur Imagenette, tanpa dependensi ML."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "imagenette2-160"
# Jumlah gambar pada arsip resmi; salinan tambahan '(1).JPEG' bukan sampel baru.
EXPECTED_COUNTS = {"train": 9469, "val": 3925}
CLASS_IDS = (
    "n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
    "n03394916", "n03417042", "n03425413", "n03445777", "n03888257",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}


def resolve_data_dir(data_dir=None):
    """Path kustom relatif terhadap terminal; default relatif terhadap proyek."""
    return DEFAULT_DATA_DIR if data_dir is None else Path(data_dir).expanduser().resolve()


def validate_dataset(data_dir=None, *, include_test=True):
    """Periksa 10 kelas dan jumlah gambar resmi; tidak memeriksa isi setiap gambar."""
    data_dir = resolve_data_dir(data_dir)
    counts = {}
    for split, expected in EXPECTED_COUNTS.items():
        if split == "val" and not include_test:
            continue
        split_dir = data_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Folder dataset tidak ditemukan: {split_dir}. "
                "Jalankan python -m utils.download_data dari root proyek."
            )
        class_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
        if [path.name for path in class_dirs] != list(CLASS_IDS):
            raise ValueError(f"Folder {split_dir} harus berisi 10 kelas Imagenette resmi.")
        per_class = [
            sum(path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                for path in class_dir.rglob("*"))
            for class_dir in class_dirs
        ]
        counts[split] = sum(per_class)
        if not all(per_class) or counts[split] != expected:
            raise ValueError(
                f"Dataset belum lengkap: {split_dir} berisi {counts[split]} gambar; "
                f"seharusnya {expected}, dengan gambar pada setiap kelas."
            )
    return counts
