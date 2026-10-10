#!/usr/bin/env bash
# Run only inside an admitted remote allocation; retain the entire result directory.
set -euo pipefail
run_dir=${1:?usage: run_hopper_codec.sh NEW_RESULT_DIRECTORY}
mkdir "$run_dir"
run_dir=$(cd "$run_dir" && pwd)
cd "$(dirname "$0")"

# Compilation is separate: make the codec with GPU_ARCH=sm_90, then test_split_float.
./test_split_float --geometry > "$run_dir/geometry.jsonl" 2> "$run_dir/geometry.stderr"
./test_split_float > "$run_dir/roundtrip.jsonl" 2> "$run_dir/roundtrip.stderr"
compute-sanitizer --tool memcheck --error-exitcode 86 ./test_split_float --smoke \
  > "$run_dir/memcheck.stdout" 2> "$run_dir/memcheck.stderr"
for run in 1 2 3; do
  sizes=(8388608 67108864)
  if (( run % 2 == 0 )); then sizes=(67108864 8388608); fi
  for bytes in "${sizes[@]}"; do
    ./test_split_float --benchmark "$bytes" 30 \
      > "$run_dir/bf16-${bytes}-${run}.jsonl" \
      2> "$run_dir/bf16-${bytes}-${run}.stderr"
  done
done
printf '%s\n' 'PASS: native codec correctness, memcheck and baseline sampling' \
  > "$run_dir/native-codec.status"
