"""Regresi pemilihan checkpoint, metrik, agregasi, dan hasil eksperimen."""

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

import train
from utils.paths import CLASS_IDS, PROJECT_ROOT
from utils.training import aggregate_results, compute_metrics, is_better_checkpoint


class TinyDataset(TensorDataset):
    def __init__(self, data_dir, length):
        labels = torch.arange(length) % 2
        super().__init__(torch.randn(length, 3, 8, 8), labels)
        self.classes = list(CLASS_IDS)
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.samples = [(str(data_dir / "val" / self.classes[int(label)] / f"{index}.JPEG"), int(label))
                        for index, label in enumerate(labels)]


class TrainingTests(unittest.TestCase):
    def setUp(self):
        temporary_root = PROJECT_ROOT / "tmp"
        temporary_root.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="training-test-", dir=temporary_root)
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(PROJECT_ROOT))
        self.addCleanup(self.temporary.cleanup)

    def test_validation_accuracy_has_priority_and_loss_breaks_ties(self):
        self.assertTrue(is_better_checkpoint(81, 0.9, 80, 0.2))
        self.assertFalse(is_better_checkpoint(79, 0.1, 80, 0.2))
        self.assertTrue(is_better_checkpoint(80, 0.1, 80, 0.2))
        self.assertFalse(is_better_checkpoint(80, 0.2, 80, 0.2))

    def test_macro_f1_includes_classes_without_predictions(self):
        result = compute_metrics([0, 0, 1, 1], [0, 1, 1, 1], ["a", "b", "c"])
        self.assertEqual(result["accuracy_percent"], 75.0)
        self.assertAlmostEqual(result["f1_macro"], ((2 / 3) + 0.8) / 3)
        self.assertEqual(result["confusion_matrix"], [[1, 1, 0], [0, 2, 0], [0, 0, 0]])

    def test_summary_uses_sample_std_and_does_not_invent_single_seed_std(self):
        rows = [{"accuracy_percent": 50.0, "f1_macro": 0.5}, {"accuracy_percent": 70.0, "f1_macro": 0.7}]
        result = aggregate_results(rows)
        self.assertEqual(result["accuracy_percent"]["mean"], 60.0)
        self.assertAlmostEqual(result["accuracy_percent"]["std"], math.sqrt(200))
        self.assertIsNone(aggregate_results(rows[:1])["accuracy_percent"]["std"])
        self.assertIsNone(aggregate_results([])["f1_macro"]["mean"])

    def test_defaults_keep_the_original_experiment(self):
        args = train.parse_args([])
        self.assertEqual((args.epochs, args.batch_size, args.lr, args.seeds), (30, 64, 0.01, [0, 1, 2]))

    def fixtures(self):
        data_dir = self.root / "fixture-data"
        source = TinyDataset(data_dir, 6)
        evaluation = TinyDataset(data_dir, 4)
        loaders = (DataLoader(Subset(source, [0, 1, 2, 3]), batch_size=2),
                   DataLoader(Subset(source, [4, 5]), batch_size=2),
                   DataLoader(evaluation, batch_size=2))
        args = train.parse_args(["--epochs", "2", "--batch-size", "2", "--seeds", "7",
                                 "--device", "cpu", "--num-workers", "0", "--data-dir", str(data_dir),
                                 "--output-dir", str(self.root), "--run-name", "fixture-run"])
        return args, loaders

    def test_final_evaluation_reloads_best_epoch_and_preserves_existing_runs(self):
        args, loaders = self.fixtures()
        actual_evaluate = train.evaluate
        validation_calls = []
        final_calls = []
        run_dir = self.root / "fixture-run"

        def controlled_evaluate(model, loader, criterion, device, *, collect_predictions=False):
            if loader is loaders[1]:
                validation_calls.append(True)
                return {"loss": 0.3 if len(validation_calls) == 1 else 0.8,
                        "accuracy_percent": 90.0 if len(validation_calls) == 1 else 10.0, "n_samples": 2}
            self.assertTrue(collect_predictions)
            best = torch.load(run_dir / "seed_7/best.pt", weights_only=True)
            last = torch.load(run_dir / "seed_7/last.pt", weights_only=True)
            self.assertEqual(best["epoch"], 1)
            self.assertEqual(last["epoch"], 2)
            self.assertTrue(any(not torch.equal(best["model_state_dict"][key], last["model_state_dict"][key])
                                for key in best["model_state_dict"]))
            for key, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, best["model_state_dict"][key]))
            final_calls.append(True)
            return actual_evaluate(model, loader, criterion, device, collect_predictions=True)

        with patch("train.validate_dataset", return_value={"train": 6, "val": 4}), \
             patch("train.get_dataloaders", return_value=loaders), \
             patch("train.environment_snapshot", return_value={}), \
             patch("train.CBAMResNet18", side_effect=lambda num_classes: torch.nn.Sequential(
                 torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, num_classes))), \
             patch("train.evaluate", side_effect=controlled_evaluate):
            self.assertEqual(train.run_experiment(args), run_dir)
            with self.assertRaises(FileExistsError):
                train.run_experiment(args)
        self.assertEqual(len(validation_calls), 2)
        self.assertEqual(len(final_calls), 1)
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertEqual((summary["status"], summary["n"], summary["seeds"][0]["best_epoch"]), ("completed", 1, 1))
        self.assertIsNone(summary["accuracy_percent"]["std"])
        with (run_dir / "seed_7/history.csv").open(newline="") as stream:
            history = list(csv.DictReader(stream))
        self.assertEqual([row["is_best"] for row in history], ["True", "False"])
        with (run_dir / "seed_7/predictions.csv").open(newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 4)
        for name in ("metrics.json", "classification_report.json", "confusion_matrix.csv",
                     "confusion_matrix.png", "training_curves.png", "split.json"):
            self.assertTrue((run_dir / "seed_7" / name).is_file(), name)

    def test_failed_training_records_failure_instead_of_completed_results(self):
        args, loaders = self.fixtures()
        with patch("train.validate_dataset", return_value={"train": 6, "val": 4}), \
             patch("train.get_dataloaders", return_value=loaders), \
             patch("train.environment_snapshot", return_value={}), \
             patch("train.CBAMResNet18", side_effect=lambda num_classes: torch.nn.Sequential(
                 torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, num_classes))), \
             patch("train.train_one_epoch", side_effect=RuntimeError("fixture training failure")):
            with self.assertRaisesRegex(RuntimeError, "fixture training failure"):
                train.run_experiment(args)
        summary = json.loads((self.root / "fixture-run/summary.json").read_text())
        status = json.loads((self.root / "fixture-run/seed_7/status.json").read_text())
        self.assertEqual((summary["status"], summary["n"], status["status"]), ("failed", 0, "failed"))
        self.assertIn("fixture training failure", summary["error"])


if __name__ == "__main__":
    unittest.main()
