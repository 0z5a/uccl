"""Optional verl checkpoint backend using receiver-ready UCCL writes.

Import through checkpoint_engine.custom_backend_module. GPU/RDMA execution
must be qualified before reporting an RL end-to-end result.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import pickle
import time
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import ray
import torch
import zmq
from uccl_zip_credits import ConsumerCredits
from verl.checkpoint_engine.base import (
    CheckpointEngine,
    CheckpointEngineRegistry,
    TensorMeta,
    merge_weight_chunks,
    split_weight_chunks,
)

if TYPE_CHECKING:
    from uccl import p2p


@dataclass
class ControlMetadata:
    address: str
    round: int
    bucket_size: int
    depth: int
    verify_payload: bool


@dataclass
class Hello:
    round: int
    endpoint: bytes
    destinations: list[bytes]


@dataclass
class Bucket:
    round: int
    step: int | None
    sequence: int
    slot: int
    size: int
    digest: str | None
    tensors: list[TensorMeta]


@dataclass
class Consumed:
    round: int
    sequence: int
    slot: int


@dataclass
class Finished:
    round: int
    step: int | None
    count: int


@dataclass
class Buffer:
    tensor: torch.Tensor
    registration: int | None = None
    dtype: torch.dtype | None = None
    extent: int = 0


@CheckpointEngineRegistry.register("uccl_zip")
class UCCLZipCheckpointEngine(CheckpointEngine):
    def __init__(
        self,
        bucket_size: int,
        is_master: bool = False,
        mode: Literal["raw", "zip"] = "zip",
        depth: int = 16,
        timeout: float = 600,
        verify_payload: bool = True,
    ):
        if bucket_size < 16 * 1024 * 1024 or depth < 1 or timeout <= 0:
            raise ValueError("Require a >=16 MiB bucket and positive depth/timeout")
        if mode not in ("raw", "zip") or os.environ.get("UCCL_P2P_TRANSPORT", "rdma") != "rdma":
            raise ValueError("Require raw/zip mode and the RDMA transport")
        strategy = os.environ.get("UCCL_P2P_COMPRESS_STRATEGY", "none")
        allowed = ("none", "off", "0") if mode == "raw" else ("encode", "split_encode", "full", "1")
        if strategy not in allowed:
            raise ValueError("Compression strategy does not match checkpoint mode")
        self.bucket_size, self.depth, self.timeout = bucket_size, depth, timeout
        self.is_master, self.mode = is_master, mode
        self.verify_payload = verify_payload
        self.round, self.rank = 0, -1
        self.address = ""
        self.context = zmq.Context()
        self.control: zmq.Socket
        self.endpoint: p2p.Endpoint
        self.endpoint_created = False
        self.buffers: list[Buffer] = []
        self.connections: dict[int, int] = {}
        self.peer_endpoints: dict[int, bytes] = {}
        self.destinations: dict[int, list[bytes]] = {}
        self.float_types: dict[torch.dtype, p2p.FloatType] = {}
        self.pending: dict[int, asyncio.Task[None]] = {}
        self.credit_events: dict[int, asyncio.Event] = {}
        self.finished = False

    def prepare(self) -> ControlMetadata:
        if self.buffers:
            raise RuntimeError("Previous checkpoint buffers have not been finalized")
        self.round += 1
        self.finished = False
        if self.is_master and not self.address:
            self.control = self.context.socket(zmq.ROUTER)
            ip = ray.util.get_node_ip_address().strip("[]")
            self.control.setsockopt(zmq.IPV6, int(":" in ip))
            base = f"tcp://[{ip}]" if ":" in ip else f"tcp://{ip}"
            self.address = f"{base}:{self.control.bind_to_random_port(base)}"
        return ControlMetadata(self.address, self.round, self.bucket_size, self.depth, self.verify_payload)

    @classmethod
    def build_topology(
        cls, actor_wg_world_size: int, rollout_world_size: int, metadata: list[ControlMetadata]
    ) -> tuple[dict[str, list[int] | list[ControlMetadata]], dict[str, list[int] | list[ControlMetadata]]]:
        master = metadata[0]
        if not master.address or len(metadata) != actor_wg_world_size + rollout_world_size:
            raise ValueError("Incomplete checkpoint worker metadata")
        if any(
            (m.round, m.bucket_size, m.depth, m.verify_payload)
            != (master.round, master.bucket_size, master.depth, master.verify_payload)
            for m in metadata
        ):
            raise ValueError("Checkpoint workers disagree on the transfer contract")
        actors = {
            "rank": [0] + [-1] * (actor_wg_world_size - 1),
            "world_size": [rollout_world_size + 1] * actor_wg_world_size,
            "master_metadata": [master] * actor_wg_world_size,
        }
        rollouts = {
            "rank": list(range(1, rollout_world_size + 1)),
            "world_size": [rollout_world_size + 1] * rollout_world_size,
            "master_metadata": [master] * rollout_world_size,
        }
        return actors, rollouts

    def init_process_group(self, rank: int, world_size: int, master_metadata: ControlMetadata) -> None:
        self.rank = rank
        if rank < 0:
            return
        from uccl import p2p

        if not self.endpoint_created:
            self.endpoint = p2p.Endpoint(local_gpu_idx=torch.cuda.current_device())
            self.float_types = {
                torch.float16: p2p.FloatType.kFloat16,
                torch.bfloat16: p2p.FloatType.kBFloat16,
                torch.float32: p2p.FloatType.kFloat32,
            }
            self.endpoint_created = True
            if rank > 0:
                if not self.endpoint.start_passive_accept():
                    raise RuntimeError("Starting the RDMA receiver failed")
                self.control = self.context.socket(zmq.DEALER)
                self.control.setsockopt(zmq.IDENTITY, str(rank).encode())
                self.control.setsockopt(zmq.IPV6, int("[" in master_metadata.address))
                self.control.connect(master_metadata.address)
                self.address = master_metadata.address
        if self.address != master_metadata.address:
            raise ValueError("Checkpoint control endpoint changed; recreate the engine")
        self.control.setsockopt(zmq.RCVTIMEO, int(self.timeout * 1000))
        self.buffers = [
            Buffer(torch.empty(self.bucket_size, dtype=torch.uint8, device="cuda")) for _ in range(self.depth)
        ]
        if rank > 0:
            destinations = []
            for buffer in self.buffers:
                ok, registration = self.endpoint.reg(buffer.tensor.data_ptr(), self.bucket_size)
                if not ok:
                    raise RuntimeError("Destination registration failed")
                buffer.registration = registration
                ok, blob = self.endpoint.advertise(registration, buffer.tensor.data_ptr(), self.bucket_size)
                if not ok or len(blob) != 64:
                    raise RuntimeError("Destination advertisement failed")
                destinations.append(blob)
            self.control.send_pyobj(Hello(self.round, bytes(self.endpoint.get_metadata()), destinations))
            return
        receivers = set(range(1, world_size))
        if len(receivers) > int(os.environ.get("UCCL_P2P_ZIP_MAX_PEERS", "4")):
            raise ValueError("Zip peer capacity is smaller than the rollout group")
        received: set[int] = set()
        while received != receivers:
            identity, payload = self.control.recv_multipart()
            receiver, hello = int(identity), pickle.loads(payload)
            if receiver not in receivers or receiver in received or not isinstance(hello, Hello):
                raise ValueError("Unexpected checkpoint handshake")
            if hello.round != self.round or len(hello.destinations) != self.depth:
                raise ValueError("Checkpoint handshake has a stale round or wrong depth")
            if receiver in self.connections:
                if self.peer_endpoints[receiver] != hello.endpoint:
                    raise ValueError("Checkpoint GPU endpoint changed; recreate the engine")
            else:
                ip, port, gpu = self.endpoint.parse_metadata(hello.endpoint)
                ok, connection = self.endpoint.connect(ip, gpu, remote_port=port)
                if not ok:
                    raise RuntimeError("RDMA connection failed")
                self.connections[receiver] = connection
                self.peer_endpoints[receiver] = hello.endpoint
            self.destinations[receiver] = hello.destinations
            received.add(receiver)
        self.credits = ConsumerCredits(self.depth, receivers)

    async def _readable(self) -> None:
        deadline = time.monotonic() + self.timeout
        while not self.control.poll(0):
            if time.monotonic() >= deadline:
                raise TimeoutError("Checkpoint control deadline")
            await asyncio.sleep(0)

    async def _receive_credits(self) -> None:
        while True:
            await self._readable()
            identity, payload = self.control.recv_multipart()
            credit = pickle.loads(payload)
            if not isinstance(credit, Consumed) or credit.round != self.round:
                raise ValueError("Unexpected or stale consumer credit")
            if self.credits.acknowledge(credit.sequence, credit.slot, int(identity)):
                self.credit_events[credit.sequence].set()

    def _write_bucket(self, slot: int, size: int, compressed: bool) -> None:
        buffer = self.buffers[slot]
        assert buffer.registration is not None
        write = self.endpoint.write_compressed_async if compressed else self.endpoint.write_async
        handles: list[int] = []
        successful = True
        for receiver, connection in self.connections.items():
            ok, handle = write(
                connection, buffer.registration, buffer.tensor.data_ptr(), size, self.destinations[receiver][slot]
            )
            successful &= ok
            if ok:
                handles.append(handle)
            else:
                break
        while handles:
            outstanding = []
            for handle in handles:
                ok, done = self.endpoint.poll_async(handle)
                successful &= ok
                if not done:
                    outstanding.append(handle)
            handles = outstanding
            if handles:
                time.sleep(0)
        if not successful:
            raise RuntimeError("Checkpoint write admission or completion failed")

    async def _send_bucket(
        self, bucket: Bucket, size: int, compressed: bool, credit_reader: asyncio.Task[None]
    ) -> None:
        await asyncio.to_thread(self._write_bucket, bucket.slot, size, compressed)
        payload = pickle.dumps(bucket)
        for receiver in self.connections:
            self.control.send_multipart([str(receiver).encode(), payload])
        consumed = asyncio.create_task(self.credit_events[bucket.sequence].wait())
        try:
            done, _ = await asyncio.wait(
                (consumed, credit_reader), timeout=self.timeout, return_when=asyncio.FIRST_COMPLETED
            )
            if credit_reader in done:
                credit_reader.result()
            if consumed not in done:
                raise TimeoutError("Checkpoint consumer credit deadline")
        finally:
            consumed.cancel()
            await asyncio.gather(consumed, return_exceptions=True)
        self.credits.release(bucket.sequence, bucket.slot)
        del self.credit_events[bucket.sequence]

    async def send_weights(
        self, weights: Generator[tuple[str, torch.Tensor], None, None], global_steps: int | None = None
    ) -> dict[str, float]:
        if self.rank < 0:
            async for _ in split_weight_chunks(weights, self.bucket_size):
                pass
            return {}
        assert self.rank == 0
        before = {peer: self.endpoint.compressed_counters(conn) for peer, conn in self.connections.items()}
        sequence, offset, logical_bytes, compressed_buckets = 0, 0, 0, 0
        dtype: torch.dtype | None = None
        metadata: list[TensorMeta] = []
        start = time.monotonic()
        credit_reader = asyncio.create_task(self._receive_credits())

        async def publish() -> None:
            nonlocal sequence, offset, metadata, compressed_buckets
            slot = sequence % self.depth
            buffer = self.buffers[slot]
            assert dtype is not None
            if (buffer.dtype, buffer.extent) != (dtype, offset):
                if buffer.registration is not None and not self.endpoint.dereg(buffer.registration):
                    raise RuntimeError("Source deregistration failed")
                ok, registration = self.endpoint.reg(
                    buffer.tensor.data_ptr(), offset, floatType=self.float_types[dtype]
                )
                if not ok:
                    raise RuntimeError("Source registration failed")
                buffer.registration, buffer.dtype, buffer.extent = registration, dtype, offset
            compressed = self.mode == "zip" and offset >= 16 * 1024 * 1024
            compressed_buckets += int(compressed)
            torch.cuda.synchronize()
            self.credits.reserve(sequence, slot)
            self.credit_events[sequence] = asyncio.Event()
            digest = (
                hashlib.sha256(buffer.tensor[:offset].cpu().numpy().tobytes()).hexdigest()
                if self.verify_payload
                else None
            )
            bucket = Bucket(self.round, global_steps, sequence, slot, offset, digest, metadata)
            self.pending[slot] = asyncio.create_task(self._send_bucket(bucket, offset, compressed, credit_reader))
            sequence, offset, metadata = sequence + 1, 0, []
            next_slot = sequence % self.depth
            if next_slot in self.pending:
                await self.pending.pop(next_slot)

        try:
            async for meta, chunk in split_weight_chunks(weights, self.bucket_size):
                assert chunk is not None
                if meta.dtype not in self.float_types:
                    raise ValueError("Checkpoint backend supports FP16, BF16 and FP32 weights")
                if metadata and (dtype != meta.dtype or offset + meta.chunk_size > self.bucket_size):
                    await publish()
                dtype = meta.dtype
                meta.offset = offset
                self.buffers[sequence % self.depth].tensor[offset : offset + meta.chunk_size].copy_(
                    chunk, non_blocking=True
                )
                metadata.append(meta)
                offset += meta.chunk_size
                logical_bytes += meta.chunk_size
            if metadata:
                await publish()
            await asyncio.gather(*self.pending.values())
            if credit_reader.done():
                credit_reader.result()
            self.pending.clear()
            for receiver in self.connections:
                self.control.send_multipart(
                    [str(receiver).encode(), pickle.dumps(Finished(self.round, global_steps, sequence))]
                )
            self.finished = True
        finally:
            credit_reader.cancel()
            await asyncio.gather(credit_reader, return_exceptions=True)
        after = {peer: self.endpoint.compressed_counters(conn) for peer, conn in self.connections.items()}
        expected = compressed_buckets * len(self.connections)
        for name in ("submitted", "ready"):
            if sum(after[p][name] - before[p][name] for p in after) != expected:
                raise RuntimeError("Compressed path counters disagree with submitted buckets")
        if any(after[p]["failed"] != before[p]["failed"] or after[p]["inflight"] != 0 for p in after):
            raise RuntimeError("Compressed writes failed or remain outstanding")
        if self.mode == "zip" and logical_bytes and not expected:
            raise RuntimeError("This checkpoint did not exercise compressed writes")
        return {
            "uccl_zip/transfer_seconds": time.monotonic() - start,
            "uccl_zip/payload_verification": float(self.verify_payload),
            "uccl_zip/logical_bytes": float(logical_bytes * len(self.connections)),
            "uccl_zip/raw_requests": float((sequence - compressed_buckets) * len(self.connections)),
            "uccl_zip/compressed_requests": float(expected),
            "uccl_zip/archive_bytes": float(sum(after[p]["archive_bytes"] - before[p]["archive_bytes"] for p in after)),
        }

    async def _receive_chunks(self, step: int | None) -> AsyncGenerator[tuple[TensorMeta, torch.Tensor], None]:
        sequence = 0
        buffered: dict[int, Bucket] = {}
        while True:
            while sequence not in buffered:
                await self._readable()
                message = self.control.recv_pyobj()
                if not isinstance(message, (Bucket, Finished)) or (message.round, message.step) != (self.round, step):
                    raise ValueError("Unexpected checkpoint message or model version")
                if isinstance(message, Finished):
                    if message.count != sequence or buffered:
                        raise ValueError("Checkpoint ended before all buckets were consumed")
                    self.finished = True
                    return
                if (
                    message.sequence < sequence
                    or message.sequence in buffered
                    or message.slot != message.sequence % self.depth
                ):
                    raise ValueError("Duplicate or invalid checkpoint bucket")
                buffered[message.sequence] = message
            bucket = buffered.pop(sequence)
            buffer = self.buffers[bucket.slot].tensor
            if not 0 < bucket.size <= self.bucket_size:
                raise ValueError("Checkpoint bucket extent exceeds its destination")
            if self.verify_payload:
                digest = hashlib.sha256(buffer[: bucket.size].cpu().numpy().tobytes()).hexdigest()
                if digest != bucket.digest:
                    raise RuntimeError("Checkpoint destination bytes differ from the source")
            for meta in bucket.tensors:
                if meta.offset < 0 or meta.chunk_size <= 0 or meta.offset + meta.chunk_size > bucket.size:
                    raise ValueError("Checkpoint tensor exceeds its bucket extent")
                yield meta, buffer[meta.offset : meta.offset + meta.chunk_size]
            # The existing verl loader consumes/copies each yielded view before
            # advancing. Complete its GPU copies before granting reuse.
            torch.cuda.synchronize()
            self.control.send_pyobj(Consumed(self.round, sequence, bucket.slot))
            sequence += 1

    async def receive_weights(self, global_steps: int | None = None) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        assert self.rank > 0
        async for item in merge_weight_chunks(self._receive_chunks(global_steps), self.bucket_size):
            yield item

    def finalize(self) -> None:
        if self.rank < 0:
            return
        if not self.finished or (self.rank == 0 and (self.pending or self.credits.slots)):
            raise RuntimeError("Cannot release checkpoint buffers with unfinished transfers or consumers")
        for buffer in self.buffers:
            if buffer.registration is not None and not self.endpoint.dereg(buffer.registration):
                raise RuntimeError("Checkpoint deregistration failed")
        self.buffers.clear()
