# Eight-L4 model-shape GEMM + AllReduce sweep (2026-11-11)

This experiment measures BF16 row-parallel FFN down projection followed by
AllReduce on one eight-L4 host. GPUs 0–3 are on NUMA node 0 and GPUs 4–7 on
node 1; peer access is unavailable between the groups. Each GPU computes
`A[M,K] × B[K,N]` for its TP=8 shard, then receives the `M × N` sum. The four
model configurations are listed below. `N` is hidden size and `K` is FFN
intermediate size divided by eight.

| Model | `N` | `K` per GPU |
| --- | ---: | ---: |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/config.json) | 4096 | 1536 |
| [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/config.json) | 5120 | 2176 |
| [Gemma 4 31B IT](https://huggingface.co/google/gemma-4-31B-it/blob/main/config.json) | 5376 | 2688 |
| [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json) | 5120 | 3200 |
Decode uses `M=1,16,32`. Prefill uses `M=2048,4096,8192`, centered on the
larger of the 24 GB GPU defaults: [SGLang chunked prefill 4096](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/arg_groups/memory_hook.py)
and [vLLM API-server token budget 2048](https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py).

The host uses NVIDIA L4 GPUs with 23,034 MiB reported memory each, driver
580.178.04, CUDA 12.9.41, CUTLASS 4.5.0, and the project's NCCL 2.28.9
shim. The benchmark source is unchanged at commit `a6ecbcdf`. All methods
run in the same process, with `NCCL_NET=Socket` and the same input tensors.

| Method | GEMM | AllReduce |
| --- | --- | --- |
| `cublas+nccl` | cuBLAS | NCCL |
| `cublas+router` | cuBLAS | UCCL Lite CPU router |
| `fused+router` | resident CUTLASS BF16 GEMM producer | UCCL Lite CPU router, overlapped in one cooperative GPU grid |

The router uses 128 KiB packets and CPU reduction workers. Each executable
invocation has two warmup rounds and five measured rounds of ten calls per
method; method order rotates each round. The executable checks every output
element on all eight GPUs after every call, outside the timer. Each sample is
the largest CUDA-event duration among the GPUs. Two independent runs cover
all 24 shapes; the second reverses shape order. The reported value is the
median of the two per-run sample medians. Thus each method/shape has 50 checked
samples per run and 100 total, with 7,200 measured calls across the sweep.
The comparison also changes the GEMM kernel, so any fused gain cannot be
attributed solely to overlap.

## Results

The fused path has the lowest aggregate median at 23 of the 24 shapes. At
Qwen3.5-9B `M=16`, the cuBLAS + NCCL median is 0.292 ms and the fused median
is 0.298 ms; the fused path loses by 1.8%. At Qwen3.5-27B `M=16`, the two
runs disagree: NCCL versus fused is 0.320 versus 0.324 ms in run 1 and
0.315 versus 0.267 ms in run 2. That aggregate win is sensitive to run order
or transient effects. The 12 prefill shapes all favor the fused path in both
runs, with aggregate NCCL/fused speedup from 1.17× to 1.44×.

| Model, `M=4096` | cuBLAS + NCCL | cuBLAS + router | Fused + router | NCCL/fused |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-9B | 13.218 ms | 11.348 ms | 10.660 ms | 1.24× |
| Qwen3.5-27B | 16.940 ms | 14.441 ms | 12.491 ms | 1.36× |
| Gemma 4 31B IT | 18.166 ms | 15.361 ms | 13.264 ms | 1.37× |
| Qwen3-32B | 17.592 ms | 14.900 ms | 12.540 ms | 1.40× |

The raw samples are [run 1](samples_run1.csv) and [run 2](samples_run2.csv).
The [summary](summary.csv) records both run medians and their median. The
[decode figure](../../../docs/assets/fusion-8l4-model-gemm-ar-decode.svg) and
[prefill figure](../../../docs/assets/fusion-8l4-model-gemm-ar-prefill.svg)
show those summary values on linear latency axes.

## Reproduce

From `experimental/lite`, with the same local NCCL shim as the
[historical eight-GPU record](../cpu_router_8l4_20261110/README.md):

```bash
make -j2 build/uccl_lite_router_fusion_8l4 \
  CUDA_HOME=/usr/local/cuda-12.9 \
  CUTLASS_HOME=/data/zhongjie/cutlass-v4.5.0 \
  NCCL_HOME=build/nccl-2.28.9
python3 benchmarks/data/fusion_8l4_models_20261111/run.py --run 1
python3 benchmarks/data/fusion_8l4_models_20261111/run.py --run 2
python3 scripts/plot_router_fusion_8l4_models.py
```

The run script preserves completed shapes in each CSV, so an interrupted run
can be resumed. It writes one log per run with the exact command and benchmark
output for each shape.
