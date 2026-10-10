"""CPU tests of sampling and workload construction, not GPU codec results."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

# Sampling/shape logic needs no Torch installation or CUDA device.
with patch.dict(
    sys.modules,
    {"torch": SimpleNamespace(cuda=SimpleNamespace(Event=None, synchronize=None))},
):
    import benchmark


class Timeline:
    def __init__(self):
        self.now = 0.0
        self.calls = 0

    def call(self):
        self.now += [100.0, 200.0, 2.0, 7.0, 3.0][self.calls]
        self.calls += 1


class Event:
    def __init__(self, timeline):
        self.timeline = timeline
        self.time = 0.0

    def record(self):
        self.time = self.timeline.now

    def synchronize(self):
        pass

    def elapsed_time(self, end):
        return end.time - self.time


class BenchmarkTest(unittest.TestCase):
    def test_retains_every_sample_and_excludes_warmup(self):
        timeline = Timeline()
        with (
            patch.object(
                benchmark.torch.cuda, "Event", side_effect=lambda **kw: Event(timeline)
            ),
            patch.object(benchmark.torch.cuda, "synchronize"),
        ):
            samples = benchmark.measure(timeline.call, warmups=2, repeats=3)
        self.assertEqual(samples, [2.0, 7.0, 3.0])
        self.assertEqual(timeline.calls, 5)

    def test_high_batch_and_ragged_preserve_total(self):
        for batch in (1, 32, 128):
            for ragged in (False, True):
                with self.subTest(batch=batch, ragged=ragged):
                    sizes = benchmark.member_sizes(4194307, batch, ragged)
                    self.assertEqual(len(sizes), batch)
                    self.assertEqual(sum(sizes), 4194307)
                    self.assertGreater(min(sizes), 0)
                    if ragged and batch > 1:
                        self.assertGreater(max(sizes), min(sizes) * 3)

    def test_rejects_empty_members(self):
        with self.assertRaises(ValueError):
            benchmark.member_sizes(1, 128, True)


if __name__ == "__main__":
    unittest.main()
