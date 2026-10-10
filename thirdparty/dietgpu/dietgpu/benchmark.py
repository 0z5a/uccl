# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the LICENSE file.
"""Preallocated single-stream codec baseline; not the UCCL split-send path."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections.abc import Callable
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
    sizes = [total_words * weight // sum(weights) for weight in weights]
    sizes[-1] += total_words - sum(sizes)
    if min(sizes) < 1:
        raise ValueError("input must contain at least one word per batch member")
    return sizes


def run(args: argparse.Namespace, input_bytes: int, as_float: bool) -> dict:
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.dtype
    ]
    word_bytes = torch.empty((), dtype=dtype).element_size()
    if input_bytes % word_bytes:
        raise ValueError("input bytes must be divisible by the dtype size")
    torch.manual_seed(args.seed)
    sizes = member_sizes(input_bytes // word_bytes, args.batch, args.ragged)
    inputs = []
    for size in sizes:
        if args.distribution == "bits":
            tensor = torch.randint(
                256, (size * word_bytes,), device="cuda", dtype=torch.uint8
            )
            inputs.append(tensor.view(dtype))
        else:
            inputs.append(torch.randn(size, device="cuda", dtype=dtype))
    outputs = [torch.empty_like(tensor) for tensor in inputs]
    scratch = torch.empty(args.scratch_mib * 1024**2, device="cuda", dtype=torch.uint8)
    size_op = (
        torch.ops.dietgpu.max_float_compressed_output_size
        if as_float
        else torch.ops.dietgpu.max_any_compressed_output_size
    )
    rows, cols = size_op(inputs)
    compressed = torch.empty((rows, cols), device="cuda", dtype=torch.uint8)
    compressed_bytes = torch.empty(rows, device="cuda", dtype=torch.int32)
    compressed_rows = list(compressed.unbind(0))
    status = torch.empty(rows, device="cuda", dtype=torch.uint8)
    decoded_words = torch.empty(rows, device="cuda", dtype=torch.int32)

    def encode():
        return torch.ops.dietgpu.compress_data(
            as_float, inputs, args.checksum, scratch, compressed, compressed_bytes
        )

    def decode():
        return torch.ops.dietgpu.decompress_data(
            as_float,
            compressed_rows,
            outputs,
            args.checksum,
            scratch,
            status,
            decoded_words,
        )

    def validate() -> None:
        expected = sizes if as_float else [size * word_bytes for size in sizes]
        if (
            status.cpu().tolist() != [1] * rows
            or decoded_words.cpu().tolist() != expected
        ):
            raise RuntimeError("decoder status or element count mismatch")
        for source, output in zip(inputs, outputs):
            if not torch.equal(source.view(torch.uint8), output.view(torch.uint8)):
                raise RuntimeError("roundtrip changed input bits")
        encoded = compressed_bytes.cpu().tolist()
        if not all(0 < size <= cols for size in encoded):
            raise RuntimeError("encoded size exceeds the archive row capacity")

    _, _, encode_scratch = encode()
    decode_scratch = decode()
    torch.cuda.synchronize()
    validate()
    if max(encode_scratch, decode_scratch) > scratch.numel():
        raise RuntimeError(
            "scratch fallback occurred; increase --scratch-mib before timing"
        )
    encode_samples = measure(encode, args.warmups, args.repeats)
    decode_samples = measure(decode, args.warmups, args.repeats)
    validate()
    encode_ms = statistics.mean(encode_samples)
    decode_ms = statistics.mean(decode_samples)
    return {
        "schema_version": 1,
        "path": "ordinary_codec",
        "codec": "float" if as_float else "ans",
        "dtype": args.dtype,
        "seed": args.seed,
        "distribution": args.distribution,
        "batch": args.batch,
        "member_words": sizes,
        "input_bytes": input_bytes,
        "compressed_bytes": int(compressed_bytes.sum().item()),
        "checksum": args.checksum,
        "exact_bytes": True,
        "warmups": args.warmups,
        "samples": args.repeats,
        "encode_samples_ms": encode_samples,
        "decode_samples_ms": decode_samples,
        "encode_mean_ms": encode_ms,
        "decode_mean_ms": decode_ms,
        "encode_payload_GBps": input_bytes / (encode_ms * 1e6),
        "decode_payload_GBps": input_bytes / (decode_ms * 1e6),
        "scratch_reserved_bytes": scratch.numel(),
        "encode_scratch_bytes": encode_scratch,
        "decode_scratch_bytes": decode_scratch,
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
        help="total uncompressed bytes across the batch",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--ragged", action="store_true")
    parser.add_argument("--distribution", choices=("normal", "bits"), default="normal")
    parser.add_argument("--checksum", action="store_true")
    parser.add_argument("--scratch-mib", type=int, default=384)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if (
        min(args.bytes + [args.batch, args.scratch_mib, args.repeats]) <= 0
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
