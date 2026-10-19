"""Tuning harus konsisten antar kandidat dan tidak memakai data evaluasi akhir."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

import tune_lr
from utils.dataset import get_dataloaders
from utils.paths import CLASS_IDS, PROJECT_ROOT, validate_dataset
from utils.training import write_json


class TuningTests(unittest.TestCase):
    def setUp(self):
        temporary_root = PROJECT_ROOT / "tmp"
        temporary_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="tuning-test-", dir=temporary_root)
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(PROJECT_ROOT))
        self.addCleanup(self.temporary.cleanup)

    def args(self):
        return tune_lr.parse_args(["--epochs", "2", "--seed", "7", "--device", "cpu",
                                   "--num-workers", "0", "--output-dir", str(self.root),
                                   "--run-name", "fixture-search"])

    def test_training_only_dataset_works_without_a_held_out_directory(self):
        for class_name in CLASS_IDS:
            directory = self.root / "train" / class_name
            directory.mkdir(parents=True)
            for index in range(10):
                Image.new("RGB", (20, 18), (index * 20, 40, 60)).save(directory / f"{index}.JPEG")
        self.assertFalse((self.root / "val").exists())
        with patch("utils.paths.EXPECTED_COUNTS", {"train": 100, "val": 3928}):
            self.assertEqual(validate_dataset(self.root, include_test=False), {"train": 100})
            with self.assertRaises(FileNotFoundError):
                validate_dataset(self.root)
        first = get_dataloaders(self.root, batch_size=4, img_size=128, seed=7,
                                num_workers=0, include_test=False)
        repeated = get_dataloaders(self.root, batch_size=4, seed=7, num_workers=0, include_test=False)
        self.assertEqual([len(loader.dataset) for loader in first[:2]], [90, 10])
        self.assertIsNone(first[2])
        for left, right in zip(first[:2], repeated[:2]):
            self.assertEqual(list(left.dataset.indices), list(right.dataset.indices))
        images, labels = next(iter(first[0]))
        self.assertEqual(tuple(images.shape), (4, 3, 128, 128))
        self.assertTrue(torch.isfinite(images).all())
        self.assertEqual(tuple(labels.shape), (4,))
        with self.assertRaises(FileNotFoundError):
            get_dataloaders(self.root, num_workers=0)

    def test_winner_uses_accuracy_then_loss_then_lower_lr(self):
        rows = [
            {"learning_rate": 0.01, "best_val_accuracy_percent": 70, "best_val_loss": 0.1},
            {"learning_rate": 0.1, "best_val_accuracy_percent": 80, "best_val_loss": 0.4},
            {"learning_rate": 0.05, "best_val_accuracy_percent": 80, "best_val_loss": 0.4},
        ]
        self.assertEqual(tune_lr.select_best(rows)["learning_rate"], 0.05)
        rows[1]["best_val_loss"] = 0.3
        self.assertEqual(tune_lr.select_best(rows)["learning_rate"], 0.1)

    def fake_candidate(self, seen, *, fail_on=None):
        def run(args):
            seen.append(args)
            if args.lr == fail_on:
                raise RuntimeError("fixture candidate failure")
            run_dir = args.output_dir / args.run_name
            run_dir.mkdir()
            write_json(run_dir / "summary.json", {
                "status": "completed", "n": 1,
                "seeds": [{"seed": args.seeds[0], "best_epoch": 1,
                           "best_val_accuracy_percent": 80.0 if args.lr != 0.01 else 70.0,
                           "best_val_loss": 0.3, "training_seconds": 1.0}],
            })
            return run_dir
        return run

    def test_search_shares_seed_and_budget_and_saves_reusable_selection(self):
        seen = []
        with patch("tune_lr.train.run_experiment", side_effect=self.fake_candidate(seen)):
            run_dir = tune_lr.run_search(self.args())
            with self.assertRaises(FileExistsError):
                tune_lr.run_search(self.args())
        self.assertEqual([args.lr for args in seen], [0.01, 0.05, 0.1])
        self.assertTrue(all(args.validation_only and args.seeds == [7] and args.epochs == 2 for args in seen))
        self.assertTrue(all(args.batch_size == 64 and args.num_workers == 0 for args in seen))
        result = json.loads((run_dir / "results.json").read_text())
        selected = json.loads((run_dir / "selected_lr.json").read_text())
        self.assertEqual((result["status"], result["completed_candidates"]), ("completed", 3))
        self.assertEqual((selected["learning_rate"], selected["tuning_epochs"], selected["seed"]), (0.05, 2, 7))
        self.assertFalse(selected["test_evaluated"])

    def test_failed_search_preserves_partial_results_without_a_winner(self):
        with patch("tune_lr.train.run_experiment", side_effect=self.fake_candidate([], fail_on=0.05)):
            with self.assertRaisesRegex(RuntimeError, "fixture candidate failure"):
                tune_lr.run_search(self.args())
        run_dir = self.root / "fixture-search"
        result = json.loads((run_dir / "results.json").read_text())
        self.assertEqual((result["status"], result["completed_candidates"]), ("failed", 1))
        self.assertIsNone(result["selected_learning_rate"])
        self.assertFalse((run_dir / "selected_lr.json").exists())
        self.assertTrue((run_dir / "lr_01/summary.json").is_file())

    def test_rejects_duplicate_nonfinite_and_nonpositive_candidates(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for values in (["0.01", "0.01"], ["nan"], ["inf"], ["0"], ["-0.1"]):
                with self.subTest(values=values), self.assertRaises(SystemExit):
                    tune_lr.parse_args(["--learning-rates", *values])


if __name__ == "__main__":
    unittest.main()
