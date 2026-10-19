"""Latih CBAM-ResNet18 dan simpan hasil eksperimen per seed."""

import argparse
import logging
import math
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.cbam_resnet import CBAMResNet18
from utils.dataset import get_dataloaders
from utils.paths import PROJECT_ROOT, resolve_data_dir, validate_dataset
from utils.training import (
    CLASS_NAMES, append_history, environment_snapshot, is_better_checkpoint,
    plot_history, save_checkpoint, save_evaluation_artifacts, save_summary, write_json,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_one_epoch(model, dataloader, criterion, optimizer, device, *, logger=None):
    model.train()
    running_loss = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    total = 0
    for step, (images, labels) in enumerate(dataloader, 1):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Loss pelatihan non-finite pada batch {step}.")
        loss.backward()
        optimizer.step()
        running_loss += loss.detach() * images.size(0)
        correct += outputs.detach().argmax(1).eq(labels).sum()
        total += labels.size(0)
        if logger and (step % 25 == 0 or step == len(dataloader)):
            logger.info("  Batch %d/%d | train loss %.4f | acc %.2f%%", step, len(dataloader),
                        running_loss.item() / total, 100.0 * correct.item() / total)
    if total == 0:
        raise ValueError("Loader pelatihan tidak berisi sampel.")
    return running_loss.item() / total, 100.0 * correct.item() / total


@torch.inference_mode()
def evaluate(model, dataloader, criterion, device, *, collect_predictions=False):
    model.eval()
    running_loss = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    total = 0
    targets, predictions, probabilities = [], [], []
    for images, labels in dataloader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        if not torch.isfinite(loss).item() or not torch.isfinite(outputs).all().item():
            raise RuntimeError("Evaluasi menghasilkan nilai non-finite.")
        predicted = outputs.argmax(1)
        running_loss += loss * images.size(0)
        correct += predicted.eq(labels).sum()
        total += labels.size(0)
        if collect_predictions:
            targets.extend(labels.cpu().tolist())
            predictions.extend(predicted.cpu().tolist())
            probabilities.extend(outputs.softmax(1).max(1).values.cpu().tolist())
    if total == 0:
        raise ValueError("Loader evaluasi tidak berisi sampel.")
    result = {"loss": running_loss.item() / total, "accuracy_percent": 100.0 * correct.item() / total,
              "n_samples": total}
    if collect_predictions:
        result.update({"targets": targets, "predictions": predictions, "probabilities": probabilities})
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--data-dir", type=Path, help="Default: data/imagenette2-160 relatif terhadap proyek.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--run-name", help="Nama folder eksperimen baru; tidak boleh sudah ada.")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("epochs dan batch-size harus positif; num-workers harus >= 0.")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr harus berupa angka finite yang positif.")
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2**32 for seed in args.seeds):
        parser.error("seeds harus unik dan berada dalam rentang 0 hingga 2**32-1.")
    if args.run_name is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_name):
        parser.error("run-name harus diawali huruf/angka dan hanya berisi huruf, angka, titik, '_' atau '-'.")
    return args


def create_logger(run_dir):
    logger = logging.getLogger("cbam_training")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)sZ | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    formatter.converter = time.gmtime
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "run.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def run_seed(seed, config, run_dir, device, logger):
    seed_dir = run_dir / f"seed_{seed}"
    seed_dir.mkdir()
    write_json(seed_dir / "status.json", {"status": "training", "seed": seed, "epochs_completed": 0})
    start = time.perf_counter()
    try:
        set_seed(seed)
        loaders = get_dataloaders(data_dir=config["data_dir"], batch_size=config["batch_size"],
                                  img_size=config["image_size"], seed=seed, num_workers=config["num_workers"])
        train_loader, val_loader, test_loader = loaders
        classes = test_loader.dataset.classes
        if len(classes) != 10:
            raise ValueError("Eksperimen ini membutuhkan tepat 10 kelas Imagenette.")
        split = {"seed": seed, "train_indices": list(train_loader.dataset.indices),
                 "internal_val_indices": list(val_loader.dataset.indices),
                 "counts": dict(zip(("train", "internal_validation", "evaluation"),
                                    (len(loader.dataset) for loader in loaders))),
                 "class_to_idx": test_loader.dataset.class_to_idx,
                 "class_names": {name: CLASS_NAMES.get(name, name) for name in classes}}
        write_json(seed_dir / "split.json", split)
        seed_config = {**config, "seed": seed}
        write_json(seed_dir / "config.json", seed_config)
        model = CBAMResNet18(num_classes=10).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
        history = []
        best_accuracy, best_loss, best_epoch = -1.0, math.inf, None
        logger.info("Seed %d | %s | parameter %s", seed, split["counts"], f"{parameter_count:,}")
        for epoch in range(1, config["epochs"] + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            synchronize(device)
            epoch_start = time.perf_counter()
            learning_rate = optimizer.param_groups[0]["lr"]
            train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device, logger=logger)
            synchronize(device)
            train_seconds = time.perf_counter() - epoch_start
            val_start = time.perf_counter()
            validation = evaluate(model, val_loader, criterion, device)
            synchronize(device)
            val_seconds = time.perf_counter() - val_start
            improved = is_better_checkpoint(validation["accuracy_percent"], validation["loss"], best_accuracy, best_loss)
            if improved:
                best_accuracy, best_loss, best_epoch = validation["accuracy_percent"], validation["loss"], epoch
            row = {
                "epoch": epoch, "learning_rate": learning_rate, "train_loss": train_loss,
                "train_accuracy_percent": train_accuracy, "val_loss": validation["loss"],
                "val_accuracy_percent": validation["accuracy_percent"], "train_seconds": train_seconds,
                "val_seconds": val_seconds, "epoch_seconds": train_seconds + val_seconds,
                "peak_allocated_memory_mib": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
                "peak_reserved_memory_mib": torch.cuda.max_memory_reserved(device) / 1024**2 if device.type == "cuda" else None,
                "is_best": improved,
            }
            scheduler.step()
            checkpoint = {
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                "best_epoch": best_epoch, "best_val_accuracy_percent": best_accuracy, "best_val_loss": best_loss,
                "config": seed_config, "class_to_idx": split["class_to_idx"], "torch_version": str(torch.__version__),
            }
            if improved:
                save_checkpoint(seed_dir / "best.pt", checkpoint)
            save_checkpoint(seed_dir / "last.pt", checkpoint)
            append_history(seed_dir / "history.csv", row)
            history.append(row)
            write_json(seed_dir / "status.json", {"status": "training", "seed": seed,
                                                  "epochs_completed": epoch, "best_epoch": best_epoch})
            logger.info("Seed %d | epoch %d/%d | %.1fs | train %.4f / %.2f%% | val %.4f / %.2f%%%s",
                        seed, epoch, config["epochs"], row["epoch_seconds"], train_loss, train_accuracy,
                        validation["loss"], validation["accuracy_percent"], " | best" if improved else "")
        del checkpoint, optimizer, scheduler
        model.zero_grad(set_to_none=True)
        best = torch.load(seed_dir / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(best["model_state_dict"])
        write_json(seed_dir / "status.json", {"status": "evaluating", "seed": seed,
                                              "epochs_completed": config["epochs"], "best_epoch": best_epoch})
        logger.info("Seed %d | evaluasi akhir %d gambar memakai best.pt (epoch %d)",
                    seed, len(test_loader.dataset), best_epoch)
        synchronize(device)
        evaluation_start = time.perf_counter()
        evaluation = evaluate(model, test_loader, criterion, device, collect_predictions=True)
        synchronize(device)
        evaluation_seconds = time.perf_counter() - evaluation_start
        metrics = save_evaluation_artifacts(seed_dir, evaluation, classes, test_loader.dataset.samples,
                                            Path(config["data_dir"]), seed=seed, best_epoch=best_epoch)
        plot_history(history, seed_dir / "training_curves.png", seed)
        training_seconds = sum(row["epoch_seconds"] for row in history)
        result = {"seed": seed, "best_epoch": best_epoch, "best_val_accuracy_percent": best_accuracy,
                  "best_val_loss": best_loss, "test_loss": evaluation["loss"], **metrics,
                  "training_seconds": training_seconds}
        write_json(seed_dir / "metrics.json", {
            **result, "checkpoint": "best.pt", "n_evaluation_samples": evaluation["n_samples"],
            "parameter_count": parameter_count, "evaluation_seconds": evaluation_seconds,
            "seed_wall_seconds": time.perf_counter() - start,
            "checkpoint_selection": "maximum internal validation accuracy; minimum loss on ties",
        })
        write_json(seed_dir / "status.json", {"status": "completed", "seed": seed,
                                              "epochs_completed": config["epochs"], "best_epoch": best_epoch})
        logger.info("Seed %d selesai | test accuracy %.2f%% | F1-macro %.4f", seed,
                    result["accuracy_percent"], result["f1_macro"])
        return result
    except BaseException as exc:
        write_json(seed_dir / "status.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                              "seed": seed, "error": f"{type(exc).__name__}: {exc}"})
        raise


def run_experiment(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA tidak tersedia. Aktifkan GPU atau pilih --device cpu.")
    device = torch.device("cuda:0" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    data_dir = resolve_data_dir(args.data_dir)
    source_counts = validate_dataset(data_dir)
    run_name = args.run_name or datetime.now(timezone.utc).strftime("cbam-%Y%m%dT%H%M%S%fZ")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / run_name
    try:
        run_dir.mkdir()
    except FileExistsError:
        raise FileExistsError(f"Folder hasil sudah ada: {run_dir}. Pilih --run-name baru; hasil lama tidak ditimpa.") from None
    logger = create_logger(run_dir)
    config = {
        "run_name": run_name, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "CBAMResNet18", "num_classes": 10, "image_size": 128, "pretrained": False,
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr,
        "optimizer": "SGD", "momentum": 0.9, "weight_decay": 5e-4,
        "scheduler": "CosineAnnealingLR", "scheduler_t_max": args.epochs,
        "seeds": args.seeds, "num_workers": args.num_workers, "device": str(device),
        "data_dir": str(data_dir), "source_counts": source_counts, "internal_val_fraction": 0.10,
        "normalization_mean": [0.485, 0.456, 0.406], "normalization_std": [0.229, 0.224, 0.225],
        "checkpoint_selection": "maximum internal validation accuracy; minimum loss on ties",
    }
    results = []
    try:
        write_json(run_dir / "config.json", config)
        set_seed(args.seeds[0])
        write_json(run_dir / "environment.json", environment_snapshot(device))
        save_summary(run_dir, results, status="running", requested_seeds=args.seeds)
        logger.info("Perangkat: %s | hasil: %s", device, run_dir)
        for seed in args.seeds:
            results.append(run_seed(seed, config, run_dir, device, logger))
            save_summary(run_dir, results, status="running", requested_seeds=args.seeds)
        save_summary(run_dir, results, status="completed", requested_seeds=args.seeds)
        logger.info("Eksperimen selesai. Ringkasan: %s", run_dir / "summary.json")
        return run_dir
    except BaseException as exc:
        save_summary(run_dir, results, status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                     requested_seeds=args.seeds, error=f"{type(exc).__name__}: {exc}")
        logger.exception("Eksperimen berhenti; hasil yang sudah tersimpan dipertahankan.")
        raise
    finally:
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()


def main(argv=None):
    args = parse_args(argv)
    try:
        run_experiment(args)
    except KeyboardInterrupt:
        print("Pelatihan dihentikan. Periksa checkpoint dan status pada folder hasil.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Pelatihan gagal: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
