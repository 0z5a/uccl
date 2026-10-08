#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
nvcc=${NVCC:-$(command -v nvcc)}
cuda_dir=${CUDA_HOME:-$(dirname "$(dirname "$nvcc")")}
make -C "$root/thirdparty/dietgpu/dietgpu/float" -f Makefile.cuda \
  NVCC="$nvcc" GPU_ARCH=sm_90 -j2
make -C "$root/p2p/rdma/tests" test_compressed_ack test_async_decompress \
  CUDA_HOME="$cuda_dir" CXX="${CXX:-g++}" -j2
"$root/p2p/rdma/tests/test_compressed_ack"
