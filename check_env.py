"""Periksa environment dan satu langkah model, tanpa menjalankan eksperimen penuh."""

import argparse
import csv
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from utils.paths import PROJECT_ROOT, resolve_data_dir, validate_dataset


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def get_driver_info():
    executable = shutil.which("nvidia-smi")
    if executable is None and sys.platform == "win32":
        candidate = Path("C:/Windows/System32/nvidia-smi.exe")
        if candidate.is_file():
            executable = str(candidate)
    if executable is None:
        return {"available": False, "reason": "nvidia-smi tidak ditemukan"}
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        return {"available": True, "devices": [
            {"name": row[0].strip(), "driver_version": row[1].strip(),
             "total_memory_mib": int(row[2].strip())}
            for row in csv.reader(result.stdout.splitlines())
        ]}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        return {"available": False, "reason": str(exc)}


def run_checks(report, args):
    require((3, 12) <= sys.version_info[:2] < (3, 14), "Gunakan Python 3.12 atau 3.13.")
    import torch
    import torchvision
    import sklearn
    from models.cbam_resnet import CBAMResNet18
    from utils.dataset import get_dataloaders

    report["packages"] = dict(sorted(
        (distribution.metadata["Name"], distribution.version)
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    ))
    print(f"[OK] Python {platform.python_version()}, PyTorch {torch.__version__}, "
          f"torchvision {torchvision.__version__}, scikit-learn {sklearn.__version__}", flush=True)

    cuda_available = torch.cuda.is_available()
    report["cuda"] = {"available": cuda_available, "build_version": torch.version.cuda,
                      "cudnn_version": torch.backends.cudnn.version()}
    report["nvidia_driver"] = get_driver_info()
    require(not args.require_cuda or cuda_available,
            "CUDA tidak tersedia. Pastikan GPU NVIDIA aktif dan PyTorch CUDA terpasang; "
            "di Colab pilih runtime GPU, kemudian ulangi pemeriksaan.")
    device = torch.device("cuda:0" if cuda_available else "cpu")
    report["device"] = str(device)
    if cuda_available:
        properties = torch.cuda.get_device_properties(device)
        report["cuda"].update({
            "device_name": properties.name,
            "total_memory_mib": round(properties.total_memory / 1024**2, 2),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        })
        print(f"[OK] GPU {properties.name}, CUDA {torch.version.cuda}", flush=True)
    else:
        print("[INFO] Menjalankan pemeriksaan pada CPU.", flush=True)

    data_dir = resolve_data_dir(args.data_dir)
    source_counts = validate_dataset(data_dir)
    torch.manual_seed(0)
    if cuda_available:
        torch.cuda.manual_seed_all(0)
    loaders = get_dataloaders(data_dir=data_dir, batch_size=args.batch_size, seed=0, num_workers=0)
    split_names = ("train", "internal_validation", "evaluation")
    split_counts = {name: len(loader.dataset) for name, loader in zip(split_names, loaders)}
    require(split_counts == {"train": 8524, "internal_validation": 948, "evaluation": 3928},
            f"Pembagian dataset tidak sesuai rancangan: {split_counts}")
    report["dataset"] = {"path": str(data_dir), "source_counts": source_counts, "split_counts": split_counts}
    batches = []
    for name, loader in zip(split_names, loaders):
        images, labels = next(iter(loader))
        require(list(images.shape) == [args.batch_size, 3, 128, 128],
                f"Bentuk batch {name} tidak sesuai: {list(images.shape)}")
        require(list(labels.shape) == [args.batch_size], f"Bentuk label {name} tidak sesuai.")
        require(bool(torch.isfinite(images).all()), f"Batch {name} mengandung nilai non-finite.")
        require(bool(((labels >= 0) & (labels < 10)).all()), f"Label {name} berada di luar 10 kelas.")
        batches.append((images, labels))
    print(f"[OK] Dataset: {split_counts}; batch {list(batches[0][0].shape)}", flush=True)

    model = CBAMResNet18(num_classes=10).to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    images, labels = (tensor.to(device) for tensor in batches[0])
    first_parameter = next(model.parameters())
    previous = first_parameter.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    if cuda_available:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    outputs = model(images)
    require(list(outputs.shape) == [args.batch_size, 10], f"Bentuk output tidak sesuai: {list(outputs.shape)}")
    require(bool(torch.isfinite(outputs).all()), "Output model mengandung nilai non-finite.")
    loss = torch.nn.functional.cross_entropy(outputs, labels)
    require(bool(torch.isfinite(loss)), "Loss mengandung nilai non-finite.")
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    require(all(gradient is not None for gradient in gradients), "Ada parameter tanpa gradien.")
    require(bool(torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all()),
            "Gradien mengandung nilai non-finite.")
    optimizer.step()
    require(not torch.equal(first_parameter.detach(), previous), "Parameter tidak berubah setelah optimizer.step().")
    if cuda_available:
        torch.cuda.synchronize(device)
    report["smoke_test"] = {
        "input_shape": list(images.shape), "output_shape": list(outputs.shape),
        "loss": float(loss.detach().cpu()), "backward": "passed", "parameter_update": "passed",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": round(time.perf_counter() - start, 3),
    }
    if cuda_available:
        report["smoke_test"]["peak_allocated_memory_mib"] = round(torch.cuda.max_memory_allocated(device) / 1024**2, 2)
    print(f"[OK] Forward/backward dan pembaruan parameter; output {list(outputs.shape)}, "
          f"loss {report['smoke_test']['loss']:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true", help="Gagal bila GPU CUDA tidak tersedia.")
    parser.add_argument("--data-dir", type=Path, help="Lokasi Imagenette; default relatif terhadap root proyek.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch pemeriksaan singkat (default: 2).")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "environment-local.json",
                        help="Lokasi laporan pemeriksaan JSON.")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 948:
        parser.error("--batch-size harus berada di antara 1 dan 948.")
    report = {
        "purpose": "environment_smoke_test", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(), "python_executable": sys.executable,
        "platform": platform.platform(), "require_cuda": args.require_cuda,
    }
    exit_code = 0
    try:
        run_checks(report, args)
        report["status"] = "passed"
    except Exception as exc:
        report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        print(f"[GAGAL] {report['error']}", file=sys.stderr, flush=True)
        exit_code = 1
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Laporan pemeriksaan: {args.output.resolve()}", flush=True)
    except OSError as exc:
        print(f"[GAGAL] Tidak dapat menyimpan laporan: {exc}", file=sys.stderr, flush=True)
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
