import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]


class BenchmarkResultTest(unittest.TestCase):
    def test_skipped_fixture_matches_shared_contract(self):
        fixture = ROOT / "tests" / "fixtures" / "skipped_result.json"
        validator = ROOT / "scripts" / "validate_benchmark_result.py"
        completed = subprocess.run([sys.executable, str(validator), str(fixture)], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(fixture.read_text())["schema_version"], "1.0.0")

    def test_cuda_source_has_resident_and_transfer_cases(self):
        source = (ROOT / "src" / "vector_add.cu").read_text()
        self.assertIn("baseline.transfer-inclusive", source)
        self.assertIn("resident.kernel-loop-wall", source)
        self.assertIn("resident.total-inclusive", source)
        self.assertNotIn("resident.transfer-inclusive", source)
        self.assertIn("nvidia-smi --query-gpu=driver_version", source)
        self.assertIn("iterations_per_sample", source)
        self.assertIn("Validate baseline's final D2H output before", source)
        self.assertIn("seed", source)
        self.assertIn("12345", source)
        self.assertIn("cudaMemcpy(da, a", source)


if __name__ == "__main__":
    unittest.main()
