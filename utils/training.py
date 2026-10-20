"""Penyimpanan hasil, evaluasi klasifikasi, dan grafik eksperimen."""

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import statistics
import subprocess
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from utils.paths import PROJECT_ROOT

HISTORY_COLUMNS = (
    "epoch", "learning_rate", "train_loss", "train_accuracy_percent", "val_loss",
    "val_accuracy_percent", "train_seconds", "val_seconds", "epoch_seconds",
    "peak_allocated_memory_mib", "peak_reserved_memory_mib", "is_best",
)
CLASS_NAMES = {
    "n01440764": "Tench", "n02102040": "English springer", "n02979186": "Cassette player",
    "n03000684": "Chain saw", "n03028079": "Church", "n03394916": "French horn",
    "n03417042": "Garbage truck", "n03425413": "Gas pump", "n03445777": "Golf ball",
    "n03888257": "Parachute",
}


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                             encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_history(path, row):
    path = Path(path)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def is_better_checkpoint(accuracy, loss, best_accuracy, best_loss):
    """Prioritaskan akurasi validasi; loss lebih rendah memecahkan nilai seri."""
    return (accuracy, -loss) > (best_accuracy, -best_loss)


def compute_metrics(targets, predictions, classes):
    labels = list(range(len(classes)))
    return {
        "accuracy_percent": 100.0 * float(accuracy_score(targets, predictions)),
        "f1_macro": float(f1_score(targets, predictions, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels).tolist(),
        "classification_report": classification_report(
            targets, predictions, labels=labels, target_names=classes, output_dict=True, zero_division=0,
        ),
    }


def aggregate_results(rows, metric_names=("accuracy_percent", "f1_macro")):
    result = {"n": len(rows), "std_ddof": 1, "seeds": rows}
    for name in metric_names:
        values = [row[name] for row in rows]
        result[name] = {"mean": statistics.mean(values) if values else None,
                        "std": statistics.stdev(values) if len(values) > 1 else None}
    return result


def save_summary(run_dir, rows, *, status, requested_seeds, error=None, validation_only=False):
    metric_names = ("best_val_accuracy_percent", "best_val_loss") if validation_only else ("accuracy_percent", "f1_macro")
    summary = {"status": status, "requested_seeds": requested_seeds,
               "evaluation_mode": "internal_validation_only" if validation_only else "final_evaluation",
               **aggregate_results(rows, metric_names)}
    if error is not None:
        summary["error"] = error
    write_json(run_dir / "summary.json", summary)
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["seed", "best_epoch", "best_val_accuracy_percent", "best_val_loss"]
        if not validation_only:
            fields.extend(("test_loss", "accuracy_percent", "f1_macro"))
        fields.append("training_seconds")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def environment_snapshot(device):
    snapshot = {
        "python": platform.python_version(), "platform": platform.platform(), "device": str(device),
        "packages": dict(sorted((dist.metadata["Name"], dist.version)
                                for dist in importlib.metadata.distributions() if dist.metadata["Name"])),
        "cuda_build": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        snapshot["gpu"] = {"name": properties.name, "total_memory_mib": properties.total_memory / 1024**2}
        executable = shutil.which("nvidia-smi")
        if executable:
            try:
                driver = subprocess.run([executable, "--query-gpu=driver_version", "--format=csv,noheader"],
                                        capture_output=True, text=True, check=True, timeout=15)
                snapshot["gpu"]["driver_versions"] = driver.stdout.strip().splitlines()
            except (OSError, subprocess.SubprocessError):
                snapshot["gpu"]["driver_versions"] = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, check=True, timeout=15)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, check=True, timeout=15)
        snapshot.update({"git_commit": commit.stdout.strip(), "git_dirty": bool(status.stdout.strip())})
    except (OSError, subprocess.SubprocessError):
        snapshot.update({"git_commit": None, "git_dirty": None})
    snapshot["source_sha256"] = {
        name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
        for name in ("train.py", "tune_lr.py", "models/cbam_resnet.py", "utils/dataset.py", "utils/paths.py", "utils/training.py")
    }
    return snapshot


def _pyplot():
    # Colab dapat mewariskan backend inline yang tidak terpasang pada venv proyek.
    # Atur sebelum import: Matplotlib memvalidasi MPLBACKEND saat diimpor.
    os.environ["MPLBACKEND"] = "Agg"
    if "MPLCONFIGDIR" not in os.environ:
        cache_dir = PROJECT_ROOT / "tmp" / "matplotlib"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache_dir)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_history(history, path, seed):
    plt = _pyplot()
    from matplotlib.ticker import MaxNLocator
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
        epochs = [row["epoch"] for row in history]
        for axis, metric, label in zip(axes, ("loss", "accuracy_percent"), ("Cross-entropy loss", "Accuracy (%)")):
            for prefix, name, color in (("train", "Train", "#176B87"), ("val", "Internal validation", "#C56A23")):
                axis.plot(epochs, [row[f"{prefix}_{metric}"] for row in history],
                          marker="o", markersize=3, linewidth=1.7, color=color, label=name)
            axis.set(xlabel="Epoch", ylabel=label, xlim=(0.5, max(epochs) + 0.5))
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(alpha=0.18)
            axis.legend(frameon=False)
        axes[0].set_ylim(bottom=0)
        axes[1].set_ylim(0, 100)
        figure.suptitle(f"CBAM-ResNet18 | seed {seed}", fontsize=13)
        figure.savefig(path, dpi=300)
        plt.close(figure)


def save_evaluation_artifacts(output_dir, evaluation, classes, samples, data_dir, *, seed, best_epoch):
    metrics = compute_metrics(evaluation["targets"], evaluation["predictions"], classes)
    write_json(output_dir / "classification_report.json", metrics["classification_report"])
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true_class / predicted_class", *classes])
        writer.writerows([name, *row] for name, row in zip(classes, metrics["confusion_matrix"]))
    if len(samples) != len(evaluation["targets"]):
        raise ValueError("Jumlah prediksi tidak cocok dengan urutan sampel evaluasi.")
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "target_index", "prediction_index", "target_class", "prediction_class", "prediction_probability"])
        for sample, target, prediction, probability in zip(
            samples, evaluation["targets"], evaluation["predictions"], evaluation["probabilities"], strict=True,
        ):
            if sample[1] != target:
                raise ValueError("Urutan label evaluasi berbeda dari urutan sampel. Gunakan shuffle=False.")
            writer.writerow([Path(sample[0]).relative_to(data_dir).as_posix(), target, prediction,
                             classes[target], classes[prediction], probability])
    plt = _pyplot()
    names = [CLASS_NAMES.get(name, name) for name in classes]
    matrix = metrics["confusion_matrix"]
    maximum = max(max(row) for row in matrix)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9}):
        figure, axis = plt.subplots(figsize=(10, 8.7), constrained_layout=True)
        image = axis.imshow(matrix, cmap="Blues", vmin=0)
        axis.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=names,
                 yticklabels=names, xlabel="Predicted class", ylabel="True class",
                 title=f"Imagenette evaluation | seed {seed} | best epoch {best_epoch}")
        plt.setp(axis.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
        for row, values in enumerate(matrix):
            for column, count in enumerate(values):
                axis.text(column, row, str(count), ha="center", va="center",
                          color="white" if count > maximum / 2 else "#202020")
        figure.colorbar(image, ax=axis, label="Images", shrink=0.83)
        figure.savefig(output_dir / "confusion_matrix.png", dpi=300)
        plt.close(figure)
    return {"accuracy_percent": metrics["accuracy_percent"], "f1_macro": metrics["f1_macro"]}
