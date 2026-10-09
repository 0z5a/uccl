#!/usr/bin/env bash
# Run the existing verl separate-async model E2E recipe. Keep output private.
set -euo pipefail
umask 077

mode=${1:?Usage: run_verl_zip_rl.sh raw|zip 32|64 PRIVATE_OUTPUT}
batch=${2:?Pass the training prompt batch size}
output=${3:?Pass a new private output directory}
: "${VERL_SOURCE_ROOT:?Set the pinned verl checkout}"
: "${MODEL_PATH:?Set the task-owned local Qwen2.5-1.5B-Instruct model directory}"
: "${TRAIN_FILES:?Set the task-owned GSM8K train parquet}"
: "${VAL_FILES:?Set the task-owned GSM8K test parquet}"
: "${UCCL_RAY_NUM_CPUS:?Set the CPU budget assigned by the run controller}"

case "$mode" in
  raw) export UCCL_P2P_COMPRESS_STRATEGY=none ;;
  zip) export UCCL_P2P_COMPRESS_STRATEGY=encode ;;
  *) exit 2 ;;
esac
case "$batch" in 32|64) ;; *) exit 2 ;; esac
expected=5b3e1f86bd39aee07ae15b6bf515198b68052486
test "$(git -C "$VERL_SOURCE_ROOT" rev-parse HEAD)" = "$expected"
git -C "$VERL_SOURCE_ROOT" diff --quiet HEAD -- verl tests/special_e2e/run_v1_separate_async.sh
mkdir "$output"
output=$(cd "$output" && pwd)
trap 'printf "%s\n" "$?" > "$output/controller.exit"' EXIT
exec > "$output/trainer.stdout" 2> "$output/trainer.stderr"

integration=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$integration:$VERL_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export UCCL_P2P_TRANSPORT=rdma UCCL_P2P_ZIP_MAX_PEERS=4
export NUM_GPUS=8 N_GPUS_TRAINING=4 N_GPUS_ROLLOUT=4 ROLLOUT_TP=2
export MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct ACTOR_STRATEGY=fsdp2
export PARAMETER_SYNC_STEP=1 PPO_MINI_BATCH_SIZE="$batch"
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-6}
export MAX_PROMPT_LENGTH=512 MAX_RESPONSE_LENGTH=512 N_RESP_PER_PROMPT=4
export VERL_FILE_LOGGER_PATH="$output/metrics.jsonl"
test "$TOTAL_TRAINING_STEPS" -ge 3

python3 - > "$output/worker-env.args" <<'PY'
import json
import os
import torch
if torch.cuda.device_count() != 8 or any("H100" not in torch.cuda.get_device_name(i) for i in range(8)):
    raise RuntimeError("This recipe requires exactly eight assigned H100 GPUs")
for key, value in sorted(os.environ.items()):
    if key.startswith("UCCL_") or key in ("PYTHONPATH", "LD_LIBRARY_PATH", "VERL_FILE_LOGGER_PATH"):
        print(f"++ray_kwargs.ray_init.runtime_env.env_vars.{key}={json.dumps(value)}")
PY
mapfile -t worker_env < "$output/worker-env.args"

cd "$VERL_SOURCE_ROOT"
bash tests/special_e2e/run_v1_separate_async.sh \
  actor_rollout_ref.rollout.checkpoint_engine.backend=uccl_zip \
  actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module=uccl_zip_verl \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=64 \
  "+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.uccl_zip={mode:$mode,depth:16,timeout:600,verify_payload:${VERIFY_PAYLOAD:-true}}" \
  actor_rollout_ref.rollout.max_num_seqs=$((batch * 2)) \
  "trainer.logger=[console,file]" \
  "trainer.experiment_name=uccl-zip-rl-$mode-b$batch" \
  "trainer.default_local_dir=$output/checkpoints" \
  "hydra.run.dir=$output/hydra" \
  +ray_kwargs.ray_init.address=local \
  "ray_kwargs.ray_init.num_cpus=$UCCL_RAY_NUM_CPUS" \
  "${worker_env[@]}"
