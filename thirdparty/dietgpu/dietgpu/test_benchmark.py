"""CPU tests of sampling and workload construction, not GPU codec results."""

import sys
import unittest
from contextlib import contextmanager
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


class QueueStream:
    def __init__(self):
        self.time = 0.0

    def wait_event(self, event):
        self.time = max(self.time, event.time)


class QueueEvent:
    def __init__(self, cuda):
        self.cuda = cuda
        self.time = 0.0

    def record(self, stream=None):
        self.time = (stream or self.cuda.current_stream()).time

    def synchronize(self):
        self.cuda.event_waits += 1

    def elapsed_time(self, end):
        return end.time - self.time


class QueueCuda:
    def __init__(self):
        self.active = QueueStream()
        self.device_waits = 0
        self.event_waits = 0
        self.events_created = 0

    def current_stream(self):
        return self.active

    def Stream(self):
        return QueueStream()

    def Event(self, **kwargs):
        self.events_created += 1
        return QueueEvent(self)

    @contextmanager
    def stream(self, stream):
        previous = self.active
        self.active = stream
        yield
        self.active = previous

    def synchronize(self):
        self.device_waits += 1


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

    def test_high_concurrency_keeps_payload_and_scratch_budgets(self):
        for concurrency in (1, 8, 32, 64, 128):
            for ragged in (False, True):
                with self.subTest(concurrency=concurrency, ragged=ragged):
                    sizes = benchmark.workload_sizes(4194307, 128, concurrency, ragged)
                    self.assertEqual(sum(map(sum, sizes)), 4194307)
                    self.assertEqual(len(sizes), concurrency)
                    self.assertTrue(
                        all(len(row) == 128 and min(row) > 0 for row in sizes)
                    )
                    scratch = [
                        n * 256
                        for n in benchmark.member_sizes(384 * 4096, concurrency, False)
                    ]
                    self.assertEqual(sum(scratch), 384 * 1024**2)
                    self.assertTrue(all(n > 0 and n % 256 == 0 for n in scratch))
        with self.assertRaises(ValueError):
            benchmark.workload_sizes(4095, 128, 32, False)

    def test_wave_joins_slowest_request_before_reusing_buffers(self):
        cuda = QueueCuda()
        durations = (
            [100.0, 2.0, 5.0, 3.0],
            [200.0, 7.0, 1.0, 4.0],
            [150.0, 3.0, 6.0, 2.0],
        )
        starts = [[] for _ in durations]

        def request(index):
            def call():
                stream = cuda.current_stream()
                starts[index].append(stream.time)
                stream.time += durations[index][len(starts[index]) - 1]

            return call

        with patch.object(benchmark.torch, "cuda", cuda):
            wave = benchmark.make_wave([request(i) for i in range(3)])
            samples = benchmark.measure(wave, warmups=1, repeats=3)
        self.assertEqual(samples, [7.0, 6.0, 4.0])
        self.assertEqual(starts, [[0.0, 200.0, 207.0, 213.0]] * 3)
        self.assertEqual(cuda.device_waits, 1)
        self.assertEqual(cuda.event_waits, 1)
        self.assertEqual(cuda.events_created, 4 + 2 * 3)

    def test_single_request_retains_direct_sampling(self):
        call = Timeline().call
        self.assertIs(benchmark.make_wave([call]), call)


if __name__ == "__main__":
    unittest.main()
