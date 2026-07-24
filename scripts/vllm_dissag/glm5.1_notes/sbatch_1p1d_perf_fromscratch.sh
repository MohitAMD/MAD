#!/bin/bash
# =============================================================================
# 1P1D (EP8) disagg PERF sweep for GLM-5.1-FP8 on the FROM-SCRATCH image
# (glm5.1-fp8-disagg:mi300x-fromscratch, id af5d3ad886cc -- built entirely from
# source per GLM5.1-FP8.disagg.MI300X.from-scratch.Dockerfile; Patch A + Patch B
# baked in => GLM_SKIP_PATCHERS=1). Same image as accuracy job 206023.
# Config parity with the aiter0118 perf wrappers (long_context via run_one_config),
# with concurrencies 8 and 32 added to every cell.
#
# Grid (two passes in ONE bring-up alloc):
#   pass 1: 8000/1000, 4000/4000, 8000/4000  x  con {8,32,64,128,256}
#   pass 2: 96000/32000                       x  con {8,32,64,128}
#
# LEAF-PINNED: submit with -w of 2 nodes in ONE leaf (RDMA locality). Example:
#   sbatch -p amd-rccl -N 2 --gres=gpu:8 --time=24:00:00 --requeue \
#     -w <leafNodeA>,<leafNodeB> \
#     --job-name=glm-1p1d-perf-fromscratch --export=ALL \
#     glm5.1_notes/sbatch_1p1d_perf_fromscratch.sh
# =============================================================================
#SBATCH --job-name=glm_1p1d_perf_fromscratch
#SBATCH --partition=amd-rccl
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
# Same-leaf placement (REQUIRED for disagg KV transfer): allocate within ONE leaf switch.
#SBATCH --switches=1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/shared_inference/mdeopuja/model_blog_logs/sbatch-%j.out
#SBATCH --error=/shared_inference/mdeopuja/model_blog_logs/sbatch-%j.err

set -u
REPO=/home/mdeopuja/cohere/MAD/scripts/vllm_dissag
cd "$REPO" || { echo "cannot cd to $REPO"; exit 1; }

# ---- image / deployment env (FROM-SCRATCH image, same as job 206023) ---------
export DOCKER_IMAGE_NAME=glm5.1-fp8-disagg:mi300x-fromscratch
export IMAGE="$DOCKER_IMAGE_NAME"
export WANT_IMAGE_ID="${WANT_IMAGE_ID:-af5d3ad886cc}"
export IMAGE_TAR=/shared_inference/mdeopuja/model_blog_logs/docker_images/glm5.1-fp8-disagg-mi300x-fromscratch.tar
export GLM_SKIP_PATCHERS=1     # from-scratch image already bakes Patch A + Patch B
export SKIP_RUNTIME_PATCH=1
export PROXY_TYPE=vllm_router
export VLLM_GCN_ARCH=gfx942
export AITER_BATON_TIMEOUT=1800
export LOG_WAIT_TIMEOUT_SECONDS=9000
export MODEL_LOCAL_DIR=/shared_inference/models_blog
# high-concurrency MoRIIO KV-write headroom (deferred/transfer timeouts)
export MORIIO_DEFER_TIMEOUT="${MORIIO_DEFER_TIMEOUT:-600}"
export VLLM_MORIIO_DEFERRED_TIMEOUT_S="${VLLM_MORIIO_DEFERRED_TIMEOUT_S:-600}"
export VLLM_MORIIO_TRANSFER_TIMEOUT_S="${VLLM_MORIIO_TRANSFER_TIMEOUT_S:-600}"

echo "=== staging image on all nodes (job $SLURM_JOB_ID) ==="
srun --overlap --ntasks-per-node=1 bash -c '
  have=$(docker images -q "'"$IMAGE"'" 2>/dev/null | head -c12)
  if [ "$have" != "'"$WANT_IMAGE_ID"'" ]; then
    echo "[$(hostname)] have=[$have] want='"$WANT_IMAGE_ID"'; loading from tar..."
    docker load -i "'"$IMAGE_TAR"'" >/dev/null 2>&1 && echo "[$(hostname)] loaded=$(docker images -q "'"$IMAGE"'" | head -c12)" || echo "[$(hostname)] LOAD_FAILED"
  else echo "[$(hostname)] image OK ($have)"; fi'

echo "=== pre-launch cleanup + stale aiter lock scrub (job $SLURM_JOB_ID) ==="
srun --overlap --ntasks-per-node=1 bash -c '
  ids=$(docker ps -aq --filter name=container_GLM-5.1-FP8 2>/dev/null); [ -n "$ids" ] && docker rm -f $ids >/dev/null 2>&1
  ids=$(docker ps -aq --filter name=glm_perf 2>/dev/null); [ -n "$ids" ] && docker rm -f $ids >/dev/null 2>&1
  for p in 36473 36367 13345 8405 61005 61555 9711 30000; do fuser -k -n tcp "$p" >/dev/null 2>&1 || true; done
  docker run --rm -v /tmp:/tmp --entrypoint bash "'"$IMAGE"'" -c "rm -f /tmp/vllm_cache*/aiter_jit/build/lock_* /tmp/vllm_cache*/*/aiter_jit/build/lock_* 2>/dev/null; true" >/dev/null 2>&1 || true
  echo "[$(hostname)] cleanup done"'

echo "=== preflight --health (job $SLURM_JOB_ID) ==="
bash glm5.1_notes/preflight_stage.sh --health || { echo "preflight failed; aborting." >&2; exit 1; }

# ---- pass 1: main shapes @ con 8/32/64/128/256 ------------------------------
echo "=== 1P1D perf pass 1: 8k/1k,4k/4k,8k/4k @ 8/32/64/128/256 (job $SLURM_JOB_ID) ==="
export BENCHMARK_COMBINATIONS="8000/1000 4000/4000 8000/4000"
export BENCHMARK_CON="8 32 64 128 256"
bash glm5.1_notes/run_one_config.sh 1 1 perf_main_1p1d_fromscratch long_context

# ---- pass 2: 96k/32k @ con 8/32/64/128 -------------------------------------
echo "=== 1P1D perf pass 2: 96k/32k @ 8/32/64/128 (job $SLURM_JOB_ID) ==="
export BENCHMARK_COMBINATIONS="96000/32000"
export BENCHMARK_CON="8 32 64 128"
bash glm5.1_notes/run_one_config.sh 1 1 perf_96k_1p1d_fromscratch long_context

echo "=== DONE 1P1D perf sweep (job $SLURM_JOB_ID); results in /shared_inference/mdeopuja/model_blog_logs/${SLURM_JOB_ID}/ ==="
