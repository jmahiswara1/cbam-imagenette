"""Grafik eksperimen harus berhasil di venv yang menerima backend inline Colab."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from utils.paths import PROJECT_ROOT


class PlottingTests(unittest.TestCase):
    def test_saves_png_when_parent_requests_unavailable_inline_backend(self):
        temporary_root = PROJECT_ROOT / "tmp"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="plotting-test-", dir=temporary_root) as temporary:
            directory = Path(temporary).resolve()
            self.assertTrue(directory.is_relative_to(PROJECT_ROOT))
            environment = os.environ.copy()
            environment["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
            environment["MPLCONFIGDIR"] = str(directory / "matplotlib-cache")
            output = directory / "training_curves.png"
            script = """
import sys
from utils.training import plot_history

history = [dict(epoch=1, train_loss=2.0, val_loss=2.1,
                train_accuracy_percent=20.0, val_accuracy_percent=18.0)]
plot_history(history, sys.argv[1], seed=0)
import matplotlib
assert matplotlib.get_backend().lower() == 'agg', matplotlib.get_backend()
print('plot saved with Agg')
"""
            result = subprocess.run([sys.executable, "-X", "utf8", "-c", script, str(output)],
                                    cwd=PROJECT_ROOT, env=environment, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("plot saved with Agg", result.stdout)
            self.assertTrue(output.is_file())
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
