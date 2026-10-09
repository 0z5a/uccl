"""Distributed compressed RDMA stage; this is not a model E2E result.

Use an even world size: first half writes to every peer in the second half.
Each C worker owns Q distinct inputs and destinations, with exact BF16 bits.
Raw and Zip cases run in separate processes with strategy off / encode.
Endpoint metadata and unfiltered runtime logs must remain private.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Literal, TypedDict

import torch
import torch.distributed as dist
from uccl import p2p

Key = tuple[int, int, int]
CounterName = Literal["submitted", "ready", "failed", "archive_bytes", "inflight"]
COUNTER_NAMES: tuple[CounterName, ...] = (
    "submitted", "ready", "failed", "archive_bytes", "inflight"
)


class Counters(TypedDict):
    submitted: int
    ready: int
    failed: int
    archive_bytes: int
    inflight: int


class Summary(TypedDict):
    workload_kind: str
    mode: str
    bytes: int
    concurrency: int
    requested_depth: int
    h100_count: int
    transfer_seconds: float
    stage_seconds: float
    verified_destinations: int
    counters: Counters


def pattern(size: int, seed: int) -> torch.Tensor:
    words = torch.arange(size // 2, dtype=torch.int32, device="cuda")
    bits = ((words * 1103515245 + seed * 12345) & 0xFFFF).to(torch.int16)
    return bits.view(torch.bfloat16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("raw", "zip"), required=True)
    parser.add_argument("--bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if args.bytes <= 0 or args.bytes % 2 or args.concurrency < 1 or args.depth < 1:
        raise ValueError("Require positive BF16 byte extent, concurrency and depth")
    strategy = os.environ.get("UCCL_P2P_COMPRESS_STRATEGY", "none")
    if args.mode == "zip" and strategy not in ("encode", "split_encode", "full", "1"):
        raise ValueError("Zip case requires the full split/encode strategy")
    if args.mode == "raw" and strategy not in ("none", "off", "0"):
        raise ValueError("Raw case requires compression disabled")
    if os.environ.get("UCCL_P2P_TRANSPORT", "rdma") != "rdma":
        raise ValueError("This stage requires RDMA")

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if "H100" not in torch.cuda.get_device_name():
        raise ValueError("This qualification requires assigned H100 GPUs")
    dist.init_process_group("gloo", timeout=timedelta(seconds=args.timeout))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world < 2 or world % 2:
        raise ValueError("Require equal writer and receiver groups")
    peers = world // 2
    if args.mode == "zip" and int(os.environ.get("UCCL_P2P_ZIP_MAX_PEERS", "4")) < peers:
        raise ValueError("Zip peer capacity must cover the receiver group")
    endpoint = p2p.Endpoint(local_gpu_idx=torch.cuda.current_device())
    metadata: list[bytes] = [b""] * world
    dist.all_gather_object(metadata, bytes(endpoint.get_metadata()))
    connections: dict[int, int] = {}
    if rank < peers:
        for receiver in range(peers, world):
            ip, port, gpu = p2p.Endpoint.parse_metadata(metadata[receiver])
            ok, connection = endpoint.connect(ip, gpu, remote_port=port)
            assert ok, "RDMA connect failed"
            connections[receiver] = connection
    else:
        for _ in range(peers):
            ok, _, _, _ = endpoint.accept()
            assert ok, "RDMA accept failed"

    inputs: dict[tuple[int, int], tuple[torch.Tensor, int]] = {}
    outputs: dict[Key, torch.Tensor] = {}
    registrations: list[int] = []
    advertised: dict[Key, bytes] = {}
    for sender in range(peers):
        for worker in range(args.concurrency):
            receiver = peers + (sender + worker) % peers
            for slot in range(args.depth):
                if rank not in (sender, receiver):
                    continue
                seed = 1 + (sender * args.concurrency + worker) * args.depth + slot
                tensor = pattern(args.bytes, seed) if rank == sender else torch.empty(
                    args.bytes // 2, dtype=torch.bfloat16, device="cuda"
                )
                float_type = p2p.FloatType.kBFloat16 if rank == sender else p2p.FloatType.kUndefined
                ok, registration = endpoint.reg(tensor.data_ptr(), tensor.nbytes, floatType=float_type)
                assert ok, "Memory registration failed"
                registrations.append(registration)
                if rank == sender:
                    inputs[(worker, slot)] = (tensor, registration)
                else:
                    key = (sender, worker, slot)
                    outputs[key] = tensor
                    ok, blob = endpoint.advertise(registration, tensor.data_ptr(), tensor.nbytes)
                    assert ok and len(blob) == 64
                    advertised[key] = blob
    destinations: list[dict[Key, bytes]] = [{} for _ in range(world)]
    dist.all_gather_object(destinations, advertised)
    torch.cuda.synchronize()
    dist.barrier()
    start = time.monotonic()
    deadline = start + args.timeout

    def writer(worker: int) -> int:
        receiver = peers + (rank + worker) % peers
        connection = connections[receiver]
        pending: list[int] = []
        for slot in range(args.depth):
            tensor, registration = inputs[(worker, slot)]
            metadata_blob = destinations[receiver][(rank, worker, slot)]
            if args.mode == "zip":
                ok, handle = endpoint.write_compressed_async(
                    connection, registration, tensor.data_ptr(), tensor.nbytes, metadata_blob
                )
            else:
                ok, handle = endpoint.write_async(
                    connection, registration, tensor.data_ptr(), tensor.nbytes, metadata_blob
                )
            assert ok, "Write admission/posting failed"
            pending.append(handle)
        while pending:
            remaining: list[int] = []
            for handle in pending:
                ok, done = endpoint.poll_async(handle)
                assert ok, "Decoder/transfer failed"
                if not done:
                    remaining.append(handle)
            pending = remaining
            if time.monotonic() >= deadline:
                raise TimeoutError("Receiver-ready stage deadline")
            if pending:
                time.sleep(0)
        return len(inputs) // args.concurrency

    counters: Counters = {"submitted": 0, "ready": 0, "failed": 0, "archive_bytes": 0, "inflight": 0}
    if rank < peers:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            completed = sum(executor.map(writer, range(args.concurrency)))
        assert completed == args.concurrency * args.depth
        if args.mode == "zip":
            for connection in connections.values():
                snapshot: Counters = endpoint.compressed_counters(connection)
                for name in COUNTER_NAMES:
                    counters[name] += snapshot[name]
            assert counters["submitted"] == counters["ready"] == completed
            assert counters["failed"] == counters["inflight"] == 0
    dist.barrier()
    transfer_seconds = time.monotonic() - start
    for (sender, worker, slot), tensor in outputs.items():
        seed = 1 + (sender * args.concurrency + worker) * args.depth + slot
        expected = pattern(args.bytes, seed)
        assert torch.equal(tensor.view(torch.int16), expected.view(torch.int16)), "Destination bits differ"
    dist.barrier()
    summary: Summary = {
        "workload_kind": "rdma_stage", "mode": args.mode, "bytes": args.bytes,
        "concurrency": args.concurrency, "requested_depth": args.depth,
        "h100_count": world, "transfer_seconds": transfer_seconds,
        "stage_seconds": time.monotonic() - start, "verified_destinations": len(outputs),
        "counters": counters,
    }
    summaries: list[Summary | None] = [None] * world
    dist.all_gather_object(summaries, summary)
    if rank == 0:
        print(json.dumps({"status": "PASS", "cases": summaries}, sort_keys=True))
    for registration in registrations:
        assert endpoint.dereg(registration)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
