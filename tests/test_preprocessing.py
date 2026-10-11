from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from squarefit.preprocessing import fit_standard_scaler


class StandardScalerTests(unittest.TestCase):
    def test_fit_uses_only_training_rows_and_accepts_csv_strings(self) -> None:
        training_rows = [{"x": "1.0", "y": 10.0}, {"x": "3.0", "y": 14.0}]
        scaler = fit_standard_scaler(training_rows, ("x", "y"))
        self.assertEqual(scaler.means, (2.0, 12.0))
        self.assertAlmostEqual(scaler.standard_deviations[0], math.sqrt(2.0))
        self.assertAlmostEqual(scaler.standard_deviations[1], math.sqrt(8.0))

        # 这条“测试集”样本绝不参与 fit，只使用训练集得到的统计量转换。
        transformed_test = scaler.transform_row({"x": 10.0, "y": 20.0})
        self.assertAlmostEqual(transformed_test[0], 8.0 / math.sqrt(2.0))
        self.assertAlmostEqual(transformed_test[1], 8.0 / math.sqrt(8.0))

    def test_inverse_transform_returns_engineering_units(self) -> None:
        scaler = fit_standard_scaler([{"x": 1.0}, {"x": 3.0}], ("x",))
        self.assertEqual(scaler.inverse_transform_values((0.0,)), (2.0,))
        self.assertAlmostEqual(scaler.inverse_transform_values((1.0,))[0], 2.0 + math.sqrt(2.0))

    def test_constant_feature_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fit_standard_scaler([{"x": 1.0}, {"x": 1.0}], ("x",))


if __name__ == "__main__":
    unittest.main()
