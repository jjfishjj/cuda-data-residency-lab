import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from residency_model import build_plan


class ResidencyModelTest(unittest.TestCase):
    def test_transfer_math(self):
        plan = build_plan(1024, 2, 2, 20)
        self.assertAlmostEqual(plan.transfer_ms, 200)
        self.assertIn("pinned", plan.recommendation)

    def test_resident_pipeline(self):
        self.assertIn("常駐", build_plan(512, 10, 0, 20).recommendation)

    def test_rejects_invalid_input(self):
        with self.assertRaises(ValueError):
            build_plan(0, 1)


if __name__ == "__main__":
    unittest.main()
