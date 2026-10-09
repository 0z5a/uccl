"""Host-only checkpoint lifecycle tests with a simulated P2P data plane.

Uses real torch CPU tensors, ZeroMQ and the selected verl source's checkpoint
contract/chunk helpers. Ray scheduling, CUDA, compression and RDMA are simulated
or excluded. These tests are not model E2E or hardware performance evidence.
Set VERL_SOURCE_ROOT to the reviewed verl checkout.
"""

from __future__ import annotations

import ast
import asyncio
import ctypes
import importlib
import os
import struct
import sys
import threading
import time
import types
import unittest
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import torch

BUCKET = 16 * 1024 * 1024


def load_contract(root: Path) -> None:
    """Execute the actual contract definitions without importing a Ray cluster."""
    base = types.ModuleType("verl.checkpoint_engine.base")
    base.__dict__.update(torch=torch, ABC=ABC, abstractmethod=abstractmethod, dataclass=dataclass)
    packages = {name: types.ModuleType(name) for name in ("verl", "verl.checkpoint_engine")}
    for package in packages.values():
        package.__path__ = []
    sys.modules.update(packages)
    sys.modules[base.__name__] = base
    selected = ast.parse("from __future__ import annotations").body
    sources = (
        ("verl/workers/rollout/utils.py", {"ensure_async_iterator"}),
        (
            "verl/checkpoint_engine/base.py",
            {
                "TensorMeta",
                "CheckpointEngineRegistry",
                "CheckpointEngine",
                "split_weight_chunks",
                "merge_weight_chunks",
            },
        ),
    )
    for relative, names in sources:
        tree = ast.parse((root / relative).read_text())
        definitions = [
            node
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
        ]
        if {node.name for node in definitions} != names:
            raise ValueError("Reviewed verl checkpoint contract is incomplete")
        selected.extend(definitions)
    # Load only the reviewed local contract, excluding framework initialization.
    exec(compile(ast.Module(body=selected, type_ignores=[]), "verl checkpoint contract", "exec"), base.__dict__)  # noqa: S102
    ray = types.ModuleType("ray")
    ray.util = types.SimpleNamespace(get_node_ip_address=lambda: "127.0.0.1")
    sys.modules["ray"] = ray


class FloatType(IntEnum):
    kUndefined = 0
    kFloat16 = 1
    kBFloat16 = 2
    kFloat32 = 3


class SimulatedEndpoint:
    instances: ClassVar[dict[int, SimulatedEndpoint]] = {}
    corrupt_next = False
    connect_calls = 0

    def __init__(self, local_gpu_idx: int):
        self.identity = len(self.instances) + 1
        self.instances[self.identity] = self
        self.registrations: dict[int, tuple[int, int, FloatType]] = {}
        self.handles: dict[int, int] = {}
        self.counters: dict[int, dict[str, int]] = {}
        self.next_registration = self.next_handle = 1
        self.lock = threading.Lock()

    def start_passive_accept(self) -> bool:
        return True

    def get_metadata(self) -> bytes:
        return str(self.identity).encode()

    @staticmethod
    def parse_metadata(metadata: bytes) -> tuple[str, int, str]:
        return "simulated", int(metadata), "0"

    def connect(self, ip: str, gpu: str, remote_port: int) -> tuple[bool, int]:
        type(self).connect_calls += 1
        self.counters[remote_port] = {"submitted": 0, "ready": 0, "failed": 0, "archive_bytes": 0, "inflight": 0}
        return True, remote_port

    def reg(self, ptr: int, size: int, floatType: FloatType = FloatType.kUndefined) -> tuple[bool, int]:
        registration = self.next_registration
        self.next_registration += 1
        self.registrations[registration] = (ptr, size, floatType)
        return True, registration

    def advertise(self, registration: int, ptr: int, size: int) -> tuple[bool, bytes]:
        assert self.registrations[registration][:2] == (ptr, size)
        return True, struct.pack("QQ", ptr, size).ljust(64, b"\0")

    def dereg(self, registration: int) -> bool:
        assert registration not in self.handles.values(), "Source reused before simulated local completion"
        del self.registrations[registration]
        return True

    def _write(
        self, connection: int, registration: int, ptr: int, size: int, meta: bytes, compressed: bool
    ) -> tuple[bool, int]:
        assert self.registrations[registration][:2] == (ptr, size), "Incorrect registered extent"
        dst, capacity = struct.unpack("QQ", meta[:16])
        assert size <= capacity
        # Make a later short bucket ready before the first full bucket.
        if size == BUCKET:
            time.sleep(0.02)
        with self.lock:
            ctypes.memmove(dst, ptr, size)
            if type(self).corrupt_next:
                ctypes.c_ubyte.from_address(dst).value ^= 1
                type(self).corrupt_next = False
            if compressed:
                self.counters[connection]["submitted"] += 1
                self.counters[connection]["ready"] += 1
                self.counters[connection]["archive_bytes"] += size
            handle = self.next_handle
            self.next_handle += 1
            self.handles[handle] = registration
        return True, handle

    def write_async(self, connection: int, registration: int, ptr: int, size: int, meta: bytes) -> tuple[bool, int]:
        return self._write(connection, registration, ptr, size, meta, False)

    def write_compressed_async(
        self, connection: int, registration: int, ptr: int, size: int, meta: bytes
    ) -> tuple[bool, int]:
        return self._write(connection, registration, ptr, size, meta, True)

    def poll_async(self, handle: int) -> tuple[bool, bool]:
        with self.lock:
            del self.handles[handle]
        return True, True

    def compressed_counters(self, connection: int) -> dict[str, int]:
        with self.lock:
            return dict(self.counters[connection])


@unittest.skipUnless(os.environ.get("VERL_SOURCE_ROOT"), "Set VERL_SOURCE_ROOT to the reviewed checkout")
class CheckpointControlTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        load_contract(Path(os.environ["VERL_SOURCE_ROOT"]))
        simulated = types.ModuleType("uccl.p2p")
        simulated.Endpoint, simulated.FloatType = SimulatedEndpoint, FloatType
        uccl = types.ModuleType("uccl")
        uccl.p2p = simulated
        sys.modules.update({"uccl": uccl, "uccl.p2p": simulated})
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations"))
        cls.backend = importlib.import_module("uccl_zip_verl")
        torch.set_num_threads(1)

    def setUp(self):
        SimulatedEndpoint.instances.clear()
        SimulatedEndpoint.connect_calls = 0
        SimulatedEndpoint.corrupt_next = False
        self.workers = []
        original_empty = torch.empty

        def cpu_empty(*args, **kwargs):
            kwargs["device"] = "cpu"
            return original_empty(*args, **kwargs)

        self.enterContext(patch.object(torch, "empty", cpu_empty))
        self.enterContext(patch.object(torch.cuda, "current_device", return_value=0))
        self.enterContext(patch.object(torch.cuda, "synchronize"))

    async def asyncTearDown(self):
        pending = [task for worker in self.workers for task in worker.pending.values()]
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for worker in self.workers:
            worker.control.close(linger=0)
            worker.context.term()
        SimulatedEndpoint.instances.clear()

    def initialize_round(self):
        metadata = [worker.prepare() for worker in self.workers]
        actors, rollouts = self.backend.UCCLZipCheckpointEngine.build_topology(1, len(self.workers) - 1, metadata)
        for index, worker in enumerate(self.workers[1:]):
            worker.init_process_group(**{key: values[index] for key, values in rollouts.items()})
        self.workers[0].init_process_group(**{key: values[0] for key, values in actors.items()})

    @staticmethod
    def weights(step: int) -> list[tuple[str, torch.Tensor]]:
        return [
            ("large", torch.arange((BUCKET + 4096) // 4, dtype=torch.float32) + step),
            ("bf16", torch.arange(4096, dtype=torch.float32).to(torch.bfloat16) + step),
            ("fp16", torch.arange(127, dtype=torch.float16) + step),
        ]

    async def collect(self, worker, step: int, delay: float) -> dict[str, torch.Tensor]:
        received = {}
        async for name, tensor in worker.receive_weights(global_steps=step):
            await asyncio.sleep(delay)
            received[name] = tensor.clone()
        return received

    async def exercise(self, mode: str):
        strategy = "encode" if mode == "zip" else "none"
        with patch.dict(os.environ, UCCL_P2P_COMPRESS_STRATEGY=strategy, UCCL_P2P_TRANSPORT="rdma"):
            engine = self.backend.UCCLZipCheckpointEngine
            self.workers = [engine(BUCKET, is_master=True, mode=mode, depth=2, timeout=10)]
            self.workers += [engine(BUCKET, mode=mode, depth=2, timeout=10) for _ in range(2)]
        for step in range(3):
            self.initialize_round()
            weights = self.weights(step)
            results = await asyncio.gather(
                self.workers[0].send_weights((item for item in weights), global_steps=step),
                self.collect(self.workers[1], step, 0),
                self.collect(self.workers[2], step, 0.005),
            )
            for received in results[1:]:
                self.assertEqual(set(received), {name for name, _ in weights})
                for name, original in weights:
                    self.assertTrue(torch.equal(original.view(torch.uint8), received[name].view(torch.uint8)))
            metrics = results[0]
            self.assertEqual(metrics["uccl_zip/compressed_requests"], 2 if mode == "zip" else 0)
            self.assertEqual(metrics["uccl_zip/raw_requests"], 6 if mode == "zip" else 8)
            self.assertEqual(metrics["uccl_zip/payload_verification"], 1)
            self.assertEqual(metrics["uccl_zip/native_submitted"], metrics["uccl_zip/compressed_requests"])
            self.assertEqual(metrics["uccl_zip/native_ready"], metrics["uccl_zip/compressed_requests"])
            self.assertEqual(metrics["uccl_zip/native_failed"], 0)
            self.assertEqual(metrics["uccl_zip/native_inflight"], 0)
            self.assertEqual(metrics["uccl_zip/logical_bytes"], 2 * sum(tensor.nbytes for _, tensor in weights))
            self.assertFalse(self.workers[0].credits.slots)
            for worker in self.workers:
                worker.finalize()
                self.assertFalse(worker.endpoint.registrations)
        self.assertEqual(SimulatedEndpoint.connect_calls, 2, "Connections must survive successive checkpoints")

    async def test_raw_rounds_and_delayed_consumer(self):
        await self.exercise("raw")

    async def test_zip_rounds_and_delayed_consumer(self):
        await self.exercise("zip")

    async def test_corruption_holds_registered_buffers(self):
        with patch.dict(os.environ, UCCL_P2P_COMPRESS_STRATEGY="encode", UCCL_P2P_TRANSPORT="rdma"):
            engine = self.backend.UCCLZipCheckpointEngine
            self.workers = [engine(BUCKET, is_master=True, depth=2, timeout=0.3), engine(BUCKET, depth=2, timeout=0.3)]
        self.initialize_round()
        SimulatedEndpoint.corrupt_next = True
        results = await asyncio.gather(
            self.workers[0].send_weights((item for item in self.weights(0)), global_steps=0),
            self.collect(self.workers[1], 0, 0),
            return_exceptions=True,
        )
        self.assertIsInstance(results[0], TimeoutError)
        self.assertIsInstance(results[1], RuntimeError)
        self.assertIn("destination bytes differ", str(results[1]))
        for worker in self.workers:
            with self.assertRaises(RuntimeError):
                worker.finalize()
            self.assertTrue(worker.endpoint.registrations)

    async def test_stale_round_credit_reaches_sender(self):
        with patch.dict(os.environ, UCCL_P2P_COMPRESS_STRATEGY="encode", UCCL_P2P_TRANSPORT="rdma"):
            engine = self.backend.UCCLZipCheckpointEngine
            self.workers = [engine(BUCKET, is_master=True, depth=2, timeout=10), engine(BUCKET, depth=2, timeout=10)]
        self.initialize_round()
        receiver = self.workers[1]
        receiver.control.send_pyobj(self.backend.Consumed(receiver.round - 1, 0, 0))
        with self.assertRaisesRegex(ValueError, "stale consumer credit"):
            await asyncio.wait_for(self.workers[0].send_weights(iter(self.weights(0)), global_steps=0), 2)
        with self.assertRaises(RuntimeError):
            self.workers[0].finalize()
        self.assertTrue(self.workers[0].endpoint.registrations)


if __name__ == "__main__":
    unittest.main()
