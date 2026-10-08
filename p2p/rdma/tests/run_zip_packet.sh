#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
output=${1:?Pass a private result directory}
mkdir -p "$output"
binary="$root/p2p/rdma/tests/test_async_decompress"
trap 'printf "%s\n" "$?" > "$output/controller.exit"' EXIT
run_step() {
  local name=$1
  shift
  local code=0
  "$@" > "$output/$name.stdout" 2> "$output/$name.stderr" || code=$?
  printf '%s\n' "$code" > "$output/$name.exit"
  return "$code"
}
run_step smoke env UCCL_DIETGPU_SM_USAGE=0 "$binary" --smoke
run_step full env UCCL_DIETGPU_SM_USAGE=0 "$binary"
run_step sm8 env UCCL_DIETGPU_SM_USAGE=8 "$binary" --smoke
run_step sm32 env UCCL_DIETGPU_SM_USAGE=32 "$binary" --smoke
run_step memcheck env UCCL_DIETGPU_SM_USAGE=0 compute-sanitizer \
  --tool memcheck --error-exitcode 86 "$binary" --smoke
