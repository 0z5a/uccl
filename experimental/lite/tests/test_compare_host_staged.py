import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_host_staged import BACKENDS, summaries


class ReferenceSummaryTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            dict(
                collective=op,
                bytes="1024",
                in_place="0",
                backend=backend,
                repeat=str(repeat),
                iteration=str(iteration),
                slowest_rank_us=str(20 if backend == "nccl" else 40),
            )
            for op in ("all_gather", "reduce_scatter")
            for backend in ("nccl", *BACKENDS)
            for repeat in range(5)
            for iteration in range(2)
        ]

    def summarize(self, rows):
        return summaries(rows, [1024], 5, 2, 4)

    def test_complete_including_regression(self):
        result = self.summarize(self.rows)
        self.assertEqual(len(result), 8)
        for row in result:
            self.assertEqual(
                row["speedup_vs_local_nccl"], 1 if row["backend"] == "nccl" else 0.5
            )

    def test_missing_and_duplicate(self):
        for rows in (self.rows[:-1], self.rows + [self.rows[0]]):
            with self.assertRaises(ValueError):
                self.summarize(rows)

    def test_invalid_identity_placement_and_timing(self):
        for field, value in (
            ("backend", "author-thunderep"),
            ("collective", "all_reduce"),
            ("in_place", "1"),
            ("slowest_rank_us", "nan"),
            ("slowest_rank_us", "0"),
            ("bytes", "2048"),
        ):
            rows = copy.deepcopy(self.rows)
            rows[0][field] = value
            with self.assertRaises(ValueError):
                self.summarize(rows)


if __name__ == "__main__":
    unittest.main()
