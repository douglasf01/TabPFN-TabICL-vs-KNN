"""Check shortcut accounting and resume behavior without running large models."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np
from tabknn.evaluate import restricted_context_accuracy


class RestrictedContextTests(unittest.TestCase):
    def test_shortcuts_and_completed_checkpoint(self):
        X = np.array([[0.], [1.], [9.], [10.], [10.]])
        y = np.array([0, 0, 1, 0, 1])
        T = np.array([[0.], [10.], [5.]])
        truth = np.array([0, 0, 1])
        model = Mock()
        model.predict.return_value = np.array([1])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            with patch("tabknn.evaluate.make_model", return_value=model) as factory:
                result = restricted_context_accuracy("tabpfn", X, T, y, truth, 2, checkpoint_path=path)
                self.assertEqual(factory.call_count, 1)
                self.assertEqual(result["accuracy"], 1)
                for field in ["frac_one_class", "frac_constant_features", "frac_model_used"]:
                    self.assertAlmostEqual(result[field], 1 / 3)
            with patch("tabknn.evaluate.make_model", side_effect=AssertionError("Must resume")):
                resumed = restricted_context_accuracy("tabpfn", X, T, y, truth, 2, checkpoint_path=path)
                self.assertEqual(result, resumed)


if __name__ == "__main__":
    unittest.main()
