"""Bandingkan learning rate memakai validasi internal, tanpa evaluasi data akhir."""

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import train
from utils.paths import PROJECT_ROOT, resolve_data_dir
from utils.training import write_json

SELECTION_RULE = "maximum internal validation accuracy; minimum loss on ties; lowest learning rate on exact ties"
RESULT_COLUMNS = ("learning_rate", "seed", "epochs", "best_epoch", "best_val_accuracy_percent",
                  "best_val_loss", "training_seconds", "run_dir")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[0.01, 0.05, 0.1])
    parser.add_argument("--epochs", type=int, default=30, help="Anggaran epoch yang sama untuk setiap kandidat.")
    parser.add_argument("--seed", type=int, default=0, help="Seed bersama untuk split, inisialisasi, dan loader.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--run-name", help="Nama folder pencarian baru; hasil lama tidak ditimpa.")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("epochs dan batch-size harus positif; num-workers harus >= 0.")
    if not 0 <= args.seed < 2**32:
        parser.error("seed harus berada dalam rentang 0 hingga 2**32-1.")
    if (any(not math.isfinite(lr) or lr <= 0 for lr in args.learning_rates)
            or len(set(args.learning_rates)) != len(args.learning_rates)):
        parser.error("learning-rates harus unik dan berupa angka finite yang positif.")
    if args.run_name is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_name):
        parser.error("run-name harus diawali huruf/angka dan hanya berisi huruf, angka, titik, '_' atau '-'.")
    return args


def select_best(rows):
    """Nilai seri sepenuhnya memilih LR terkecil, terlepas dari urutan kandidat."""
    return min(rows, key=lambda row: (-row["best_val_accuracy_percent"], row["best_val_loss"], row["learning_rate"]))


def save_results(run_dir, rows, requested_rates, *, status, error=None):
    payload = {"status": status, "requested_learning_rates": requested_rates,
               "completed_candidates": len(rows), "selection_rule": SELECTION_RULE,
               "test_evaluated": False, "results": rows,
               "selected_learning_rate": select_best(rows)["learning_rate"] if status == "completed" else None}
    if error is not None:
        payload["error"] = error
    write_json(run_dir / "results.json", payload)
    with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def run_search(args):
    run_name = args.run_name or datetime.now(timezone.utc).strftime("lr-search-%Y%m%dT%H%M%S%fZ")
    run_dir = args.output_dir.expanduser().resolve() / run_name
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir()
    except FileExistsError:
        raise FileExistsError(f"Folder hasil sudah ada: {run_dir}. Pilih --run-name baru.") from None
    data_dir = resolve_data_dir(args.data_dir)
    config = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "learning_rates": args.learning_rates,
              "epochs_per_candidate": args.epochs, "seed": args.seed, "batch_size": args.batch_size,
              "num_workers": args.num_workers, "device": args.device, "data_dir": str(data_dir),
              "validation_only": True, "selection_rule": SELECTION_RULE,
              "protocol": "same seed, stratified split, initialization and epoch budget for every candidate"}
    rows = []
    try:
        write_json(run_dir / "config.json", config)
        save_results(run_dir, rows, args.learning_rates, status="running")
        for index, learning_rate in enumerate(args.learning_rates, 1):
            print(f"\nKandidat {index}/{len(args.learning_rates)} | LR {learning_rate:g} | seed {args.seed}", flush=True)
            candidate_args = train.parse_args([
                "--validation-only", "--lr", str(learning_rate), "--epochs", str(args.epochs),
                "--seeds", str(args.seed), "--batch-size", str(args.batch_size),
                "--num-workers", str(args.num_workers), "--device", args.device,
                "--data-dir", str(data_dir), "--output-dir", str(run_dir), "--run-name", f"lr_{index:02d}",
            ])
            candidate_dir = train.run_experiment(candidate_args)
            summary = json.loads((candidate_dir / "summary.json").read_text(encoding="utf-8"))
            if summary["status"] != "completed" or summary["n"] != 1:
                raise RuntimeError(f"Kandidat LR {learning_rate:g} belum selesai.")
            result = summary["seeds"][0]
            rows.append({"learning_rate": learning_rate, "epochs": args.epochs, **result,
                         "run_dir": candidate_dir.relative_to(run_dir).as_posix()})
            save_results(run_dir, rows, args.learning_rates, status="running")
        best = select_best(rows)
        write_json(run_dir / "selected_lr.json", {
            **best, "tuning_epochs": args.epochs, "batch_size": args.batch_size,
            "selection_rule": SELECTION_RULE, "test_evaluated": False,
        })
        save_results(run_dir, rows, args.learning_rates, status="completed")
        print(f"\nTuning selesai ({args.epochs} epoch/kandidat). LR terpilih: {best['learning_rate']:g}", flush=True)
        print(f"Perbandingan: {run_dir / 'results.csv'}", flush=True)
        return run_dir
    except BaseException as exc:
        save_results(run_dir, rows, args.learning_rates,
                     status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                     error=f"{type(exc).__name__}: {exc}")
        raise


def main(argv=None):
    args = parse_args(argv)
    try:
        run_search(args)
    except KeyboardInterrupt:
        print("Tuning dihentikan. Hasil parsial tersimpan; LR belum dipilih.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Tuning gagal: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
