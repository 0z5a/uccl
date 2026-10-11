# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the LICENSE file.
"""Preallocated codec waves with a fixed total payload and scratch budget."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch


def measure(call: Callable[[], object], warmups: int, repeats: int) -> list[float]:
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(repeats)
    ]
    for start, end in events:
        start.record()
        end.record()
    for _ in range(warmups):
        call()
    torch.cuda.synchronize()
    for start, end in events:
        start.record()
        call()
        end.record()
    events[-1][1].synchronize()
    return [start.elapsed_time(end) for start, end in events]


def member_sizes(total_words: int, batch: int, ragged: bool) -> list[int]:
    weights = [1 + i % 4 if ragged else 1 for i in range(batch)]
    total_weight = sum(weights)
    sizes = [total_words * weight // total_weight for weight in weights]
    sizes[-1] += total_words - sum(sizes)
    if min(sizes) < 1:
        raise ValueError("input must contain at least one word per batch member")
    return sizes


def workload_sizes(
    total_words: int, batch: int, concurrency: int, ragged: bool
) -> list[list[int]]:
    return [
        member_sizes(words, batch, ragged)
        for words in member_sizes(total_words, concurrency, False)
    ]


def make_wave(calls: list[Callable[[], object]]) -> Callable[[], object]:
    if len(calls) == 1:
        return calls[0]
    coordinator = torch.cuda.current_stream()
    streams = [torch.cuda.Stream() for _ in calls]
    start = torch.cuda.Event()
    done = [torch.cuda.Event() for _ in calls]
    start.record(coordinator)
    for stream, event in zip(streams, done):
        event.record(stream)

    def wave() -> None:
        start.record(coordinator)
        for call, stream, event in zip(calls, streams, done):
            stream.wait_event(start)
            with torch.cuda.stream(stream):
                call()
                event.record(stream)
        for event in done:
            coordinator.wait_event(event)

    return wave


@dataclass
class CodecRequest:
    as_float: bool
    checksum: bool
    inputs: list[torch.Tensor]
    outputs: list[torch.Tensor]
    scratch: torch.Tensor
    compressed: torch.Tensor
    compressed_bytes: torch.Tensor
    compressed_rows: list[torch.Tensor]
    status: torch.Tensor
    decoded_words: torch.Tensor
    encode_scratch: int = 0
    decode_scratch: int = 0

    def encode(self) -> None:
        _, _, required = torch.ops.dietgpu.compress_data(
            self.as_float,
            self.inputs,
            self.checksum,
            self.scratch,
            self.compressed,
            self.compressed_bytes,
        )
        self.encode_scratch = max(self.encode_scratch, required)

    def decode(self) -> None:
        required = torch.ops.dietgpu.decompress_data(
            self.as_float,
            self.compressed_rows,
            self.outputs,
            self.checksum,
            self.scratch,
            self.status,
            self.decoded_words,
        )
        self.decode_scratch = max(self.decode_scratch, required)

    def roundtrip(self) -> None:
        self.encode()
        self.decode()

    def poison(self) -> None:
        for tensor in self.outputs:
            tensor.view(torch.uint8).fill_(0xA5)
        self.status.zero_()
        self.decoded_words.zero_()

    def validate(self) -> None:
        rows, cols = self.compressed.shape
        expected = [
            tensor.numel() * (1 if self.as_float else tensor.element_size())
            for tensor in self.inputs
        ]
        if (
            self.status.cpu().tolist() != [1] * rows
            or self.decoded_words.cpu().tolist() != expected
        ):
            raise RuntimeError("decoder status or element count mismatch")
        for source, output in zip(self.inputs, self.outputs):
            if not torch.equal(source.view(torch.uint8), output.view(torch.uint8)):
                raise RuntimeError("roundtrip changed input bits")
        if not all(0 < size <= cols for size in self.compressed_bytes.cpu().tolist()):
            raise RuntimeError("encoded size exceeds the archive row capacity")
        if max(self.encode_scratch, self.decode_scratch) > self.scratch.numel():
            raise RuntimeError(
                "scratch fallback occurred; increase --scratch-mib before timing"
            )


def run(
    args: argparse.Namespace, input_bytes: int, as_float: bool
) -> dict[str, object]:
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.dtype
    ]
    word_bytes = torch.empty((), dtype=dtype).element_size()
    if input_bytes % word_bytes:
        raise ValueError("input bytes must be divisible by the dtype size")
    torch.manual_seed(args.seed)
    sizes = workload_sizes(
        input_bytes // word_bytes, args.batch, args.concurrency, args.ragged
    )
    scratch_sizes = [
        blocks * 256
        for blocks in member_sizes(args.scratch_mib * 4096, args.concurrency, False)
    ]
    if args.distribution == "bits":
        source = torch.randint(
            256, (input_bytes,), device="cuda", dtype=torch.uint8
        ).view(dtype)
    else:
        source = torch.randn(input_bytes // word_bytes, device="cuda", dtype=dtype)
    inputs = iter(source.split([size for request in sizes for size in request]))
    scratch = torch.empty(args.scratch_mib * 1024**2, device="cuda", dtype=torch.uint8)
    size_op = (
        torch.ops.dietgpu.max_float_compressed_output_size
        if as_float
        else torch.ops.dietgpu.max_any_compressed_output_size
    )
    requests = []
    for words, workspace in zip(sizes, scratch.split(scratch_sizes)):
        # Raw ANS requires aligned member starts; arbitrary ragged views do not.
        members = [next(inputs).clone() for _ in words]
        rows, cols = size_op(members)
        compressed = torch.empty((rows, cols), device="cuda", dtype=torch.uint8)
        requests.append(
            CodecRequest(
                as_float,
                args.checksum,
                members,
                [torch.empty_like(t) for t in members],
                workspace,
                compressed,
                torch.empty(rows, device="cuda", dtype=torch.int32),
                list(compressed.unbind(0)),
                torch.empty(rows, device="cuda", dtype=torch.uint8),
                torch.empty(rows, device="cuda", dtype=torch.int32),
            )
        )
    del inputs, source
    encode = make_wave([request.encode for request in requests])
    decode = make_wave([request.decode for request in requests])
    roundtrip = make_wave([request.roundtrip for request in requests])
    for request in requests:
        request.poison()
    roundtrip()
    torch.cuda.synchronize()
    for request in requests:
        request.validate()
    encode_samples = measure(encode, args.warmups, args.repeats)
    decode_samples = measure(decode, args.warmups, args.repeats)
    roundtrip_samples = measure(roundtrip, args.warmups, args.repeats)
    for request in requests:
        request.validate()
        request.poison()
    roundtrip()
    torch.cuda.synchronize()
    for request in requests:
        request.validate()
    encode_ms = statistics.mean(encode_samples)
    decode_ms = statistics.mean(decode_samples)
    roundtrip_ms = statistics.mean(roundtrip_samples)
    output_reserved_bytes = sum(
        tensor.numel() * tensor.element_size()
        for request in requests
        for tensor in request.outputs
        + [
            request.compressed,
            request.compressed_bytes,
            request.status,
            request.decoded_words,
        ]
    )
    return {
        "schema_version": 2,
        "path": "ordinary_codec",
        "codec": "float" if as_float else "ans",
        "dtype": args.dtype,
        "seed": args.seed,
        "distribution": args.distribution,
        "batch": args.batch,
        "batch_scope": "per_request",
        "concurrency_limit": args.concurrency,
        "actual_peak_inflight": None,
        "timing_scope": "closed_wave",
        "requests_per_sample": args.concurrency,
        "member_words": sizes,
        "input_bytes": input_bytes,
        "compressed_bytes": sum(int(r.compressed_bytes.sum().item()) for r in requests),
        "checksum": args.checksum,
        "exact_bytes": True,
        "warmups": args.warmups,
        "samples": args.repeats,
        "encode_samples_ms": encode_samples,
        "decode_samples_ms": decode_samples,
        "roundtrip_samples_ms": roundtrip_samples,
        "encode_mean_ms": encode_ms,
        "decode_mean_ms": decode_ms,
        "roundtrip_mean_ms": roundtrip_ms,
        "encode_payload_GBps": input_bytes / (encode_ms * 1e6),
        "decode_payload_GBps": input_bytes / (decode_ms * 1e6),
        "roundtrip_payload_GBps": input_bytes / (roundtrip_ms * 1e6),
        "scratch_reserved_bytes": scratch.numel(),
        "output_and_archive_reserved_bytes": output_reserved_bytes,
        "tensor_reserved_bytes": input_bytes + scratch.numel() + output_reserved_bytes,
        "scratch_partition_bytes": scratch_sizes,
        "encode_scratch_bytes": [r.encode_scratch for r in requests],
        "decode_scratch_bytes": [r.decode_scratch for r in requests],
        "preallocated_io_and_scratch": True,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--codec", choices=("float", "ans", "both"), default="both")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--bytes",
        type=int,
        nargs="+",
        default=[8 << 20, 64 << 20],
        help="total uncompressed bytes across all requests in a wave",
    )
    parser.add_argument(
        "--batch", type=int, default=1, help="members per codec request"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="independent codec requests per wave; total --bytes and scratch remain fixed",
    )
    parser.add_argument("--ragged", action="store_true")
    parser.add_argument("--distribution", choices=("normal", "bits"), default="normal")
    parser.add_argument("--checksum", action="store_true")
    parser.add_argument("--scratch-mib", type=int, default=384)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if (
        min(args.bytes + [args.batch, args.concurrency, args.scratch_mib, args.repeats])
        <= 0
        or args.warmups < 0
    ):
        parser.error(
            "sizes, batch and repeats must be positive; warmups must be nonnegative"
        )
    if torch.cuda.device_count() != 1:
        parser.error("expose exactly one admitted GPU through CUDA_VISIBLE_DEVICES")
    if int(os.environ.get("UCCL_DIETGPU_SM_USAGE", "0")) > 0:
        parser.error(
            "use the native split test for Green Context runs; this driver uses Torch streams"
        )
    torch.cuda.set_device(0)
    torch.ops.load_library(str(args.library.resolve(strict=True)))
    modes = [True, False] if args.codec == "both" else [args.codec == "float"]
    for size in args.bytes:
        for as_float in modes:
            print(json.dumps(run(args, size, as_float)), flush=True)


if __name__ == "__main__":
    main()
