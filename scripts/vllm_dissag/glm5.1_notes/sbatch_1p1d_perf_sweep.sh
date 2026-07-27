#!/bin/bash
# =============================================================================
# 1P1D disagg THROUGHPUT PERF SWEEP for GLM-5.1-FP8 on the baked PR#47766 batonfix
# image (vllm-disagg:glmv5.1-v0.24-pr47766-baked-batonfix), via benchmark_xPyD.sh.
# Analogue of the DeepSeek best-config perf sweep, using the proven GLM harness.
#
# Submit:
#   sbatch -p amd-rccl -N 2 --gres=gpu:8 --time=12:00:00 --requeue \
#          --job-name=glm-1p1d-perfsweep --export=ALL \
#          glm5.1_notes/sbatch_1p1d_perf_sweep.sh
# =============================================================================
#SBATCH --job-name=glm_1p1d_perfsweep
#SBATCH --partition=amd-rccl
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
# Same-leaf placement (REQUIRED for disagg KV transfer): allocate within ONE leaf switch.
#SBATCH --switches=1
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/shared_inference/mdeopuja/model_blog_logs/sbatch-perfsweep-%j.out
#SBATCH --error=/shared_inference/mdeopuja/model_blog_logs/sbatch-perfsweep-%j.err

set -u
REPO=/home/mdeopuja/cohere/MAD/scripts/vllm_dissag
cd "$REPO" || { echo "cannot cd to $REPO"; exit 1; }

export DOCKER_IMAGE_NAME=vllm-disagg:glmv5.1-v0.24-pr47766-baked-batonfix
export IMAGE="$DOCKER_IMAGE_NAME"
export WANT_IMAGE_ID="${WANT_IMAGE_ID:-aab075a8d1fe}"
export IMAGE_TAR=/shared_inference/mdeopuja/model_blog_logs/docker_images/glmv5.1-v0.24-pr47766-baked-batonfix.tar
export GLM_SKIP_PATCHERS=1
export SKIP_RUNTIME_PATCH=1
export PROXY_TYPE=vllm_router
export VLLM_GCN_ARCH=gfx942
export AITER_BATON_TIMEOUT=300
export MODEL_LOCAL_DIR=/shared_inference/models_blog
export LOG_WAIT_TIMEOUT_SECONDS=9000

# Throughput perf grid (1k1k + 8k1k, concurrency sweep). Mirrors the intent of the
# DeepSeek best20 perf datapoints for a 1P1D EP8 topology.
export BENCHMARK_COMBINATIONS="${BENCHMARK_COMBINATIONS:-1024/1024 8192/1024}"
export BENCHMARK_CON="${BENCHMARK_CON:-1 8 32 64 128}"
export WARMUPS=1 NUM_PROMPTS_FACTOR=4

echo "=== staging image on all nodes (job $SLURM_JOB_ID) ==="
srun --overlap --ntasks-per-node=1 bash -c '
  have=$(docker images -q "'"$IMAGE"'" 2>/dev/null | head -c12)
  if [ "$have" != "'"$WANT_IMAGE_ID"'" ]; then
    echo "[$(hostname)] have=[$have] want='"$WANT_IMAGE_ID"'; loading from tar..."
    docker load -i "'"$IMAGE_TAR"'" >/dev/null 2>&1 && echo "[$(hostname)] loaded=$(docker images -q "'"$IMAGE"'" | head -c12)" || echo "[$(hostname)] LOAD_FAILED"
  else
    echo "[$(hostname)] image OK ($have)"
  fi'

echo "=== pre-launch cleanup (job $SLURM_JOB_ID) ==="
srun --overlap --ntasks-per-node=1 bash -c '
  ids=$(docker ps -aq --filter name=container_GLM-5.1-FP8 2>/dev/null)
  [ -n "$ids" ] && docker rm -f $ids >/dev/null 2>&1
  ids2=$(docker ps -aq --filter name=glm_ 2>/dev/null)
  [ -n "$ids2" ] && docker rm -f $ids2 >/dev/null 2>&1
  for p in 36473 36367 13345 8405 61005 61555 9711 30000; do fuser -k -n tcp "$p" >/dev/null 2>&1 || true; done
  echo "[$(hostname)] cleanup done: $(docker ps -q | wc -l) containers left"'

echo "=== preflight --health (job $SLURM_JOB_ID) ==="
bash glm5.1_notes/preflight_stage.sh --health || {
    echo "preflight failed; aborting." >&2
    exit 1
}

echo "=== launching 1P1D GLM-5.1 perf sweep (job $SLURM_JOB_ID) ==="
bash glm5.1_notes/run_one_config.sh 1 1 perfsweep_1p1d sweep
