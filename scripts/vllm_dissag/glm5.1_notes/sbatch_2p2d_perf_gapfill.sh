#!/bin/bash
# =============================================================================
# 2P2D (EP16) disagg PERF gap-fill. FROM-SCRATCH image (glm5.1-fp8-disagg:mi300x-
# fromscratch, af5d3ad886cc). Fills the 2P2D holes vs 1P1D:
#   pass 1: 4000/4000 + 8000/4000  x  con {512}            (match 1P1D con=512)
#   pass 2: 32000/8000             x  con {32,64,128,256}  (con=8 already have)
# NUM_PROMPTS_FACTOR=2 bounds prompt volume at high concurrency. 32k/8k con>=128
# may partially load-shed at this heavy shape.
#
# Submit (leaf-pinned):
#   sbatch -p amd-rccl -N 4 --gres=gpu:8 --time=24:00:00 --requeue --switches=1 \
#     --exclude=<faulted> --job-name=glm-2p2d-perf-gapfill --export=ALL \
#     glm5.1_notes/sbatch_2p2d_perf_gapfill.sh
# =============================================================================
#SBATCH --job-name=glm_2p2d_perf_gapfill
#SBATCH --partition=amd-rccl
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --switches=1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/shared_inference/mdeopuja/model_blog_logs/sbatch-%j.out
#SBATCH --error=/shared_inference/mdeopuja/model_blog_logs/sbatch-%j.err

set -u
REPO=/home/mdeopuja/cohere/MAD/scripts/vllm_dissag
cd "$REPO" || { echo "cannot cd to $REPO"; exit 1; }

export DOCKER_IMAGE_NAME=glm5.1-fp8-disagg:mi300x-fromscratch
export IMAGE="$DOCKER_IMAGE_NAME"
export WANT_IMAGE_ID="${WANT_IMAGE_ID:-af5d3ad886cc}"
export IMAGE_TAR=/shared_inference/mdeopuja/model_blog_logs/docker_images/glm5.1-fp8-disagg-mi300x-fromscratch.tar
export GLM_SKIP_PATCHERS=1
export SKIP_RUNTIME_PATCH=1
export PROXY_TYPE=vllm_router
export VLLM_GCN_ARCH=gfx942
export AITER_BATON_TIMEOUT=1800
export LOG_WAIT_TIMEOUT_SECONDS=9000
export MODEL_LOCAL_DIR=/shared_inference/models_blog
export MORIIO_DEFER_TIMEOUT="${MORIIO_DEFER_TIMEOUT:-600}"
export VLLM_MORIIO_DEFERRED_TIMEOUT_S="${VLLM_MORIIO_DEFERRED_TIMEOUT_S:-600}"
export VLLM_MORIIO_TRANSFER_TIMEOUT_S="${VLLM_MORIIO_TRANSFER_TIMEOUT_S:-600}"
export NUM_PROMPTS_FACTOR="${NUM_PROMPTS_FACTOR:-2}"

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

echo "=== 2P2D gapfill pass 1: 4k/4k + 8k/4k @ 512 (job $SLURM_JOB_ID) ==="
export BENCHMARK_COMBINATIONS="4000/4000 8000/4000"
export BENCHMARK_CON="512"
bash glm5.1_notes/run_one_config.sh 2 2 perf_gapfill_2p2d_512 long_context

echo "=== 2P2D gapfill pass 2: 32k/8k @ 32/64/128/256 (job $SLURM_JOB_ID) ==="
export BENCHMARK_COMBINATIONS="32000/8000"
export BENCHMARK_CON="32 64 128 256"
bash glm5.1_notes/run_one_config.sh 2 2 perf_gapfill_2p2d_32k8k long_context

echo "=== DONE 2P2D perf gapfill (job $SLURM_JOB_ID); results in /shared_inference/mdeopuja/model_blog_logs/${SLURM_JOB_ID}/ ==="
