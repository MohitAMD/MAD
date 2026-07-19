#!/bin/bash
# =============================================================================
# 1P1D disagg NIAH for GLM-5.1-FP8 on the baked PR#47766 image WITH the self-healing
# aiter JIT baton (vllm-disagg:glmv5.1-v0.24-local-batonfix), via /v1/completions.
# GLM_SKIP_PATCHERS=1 (DSA/MoRIIO fixes baked in-source). The baton patch makes decode
# bring-up recover from a dead/hung aiter kernel builder instead of deadlocking forever.
#
# Submit:
#   sbatch -p amd-rccl -N 2 --gres=gpu:8 --time=12:00:00 --requeue \
#          --exclude=useocpm2m-097-119 \
#          --job-name=glm-1p1d-niah-pr47766-batonfix --export=ALL \
#          glm5.1_notes/sbatch_1p1d_niah_pr47766_batonfix_completions.sh
# =============================================================================
#SBATCH --job-name=glm_1p1d_niah_determ_probe
#SBATCH --partition=amd-rccl
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --spread-job
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/shared_inference/mdeopuja/model_blog_logs/sbatch-%j.out
#SBATCH --error=/shared_inference/mdeopuja/model_blog_logs/sbatch-%j.err

set -u
REPO=/home/mdeopuja/cohere/MAD/scripts/vllm_dissag
cd "$REPO" || { echo "cannot cd to $REPO"; exit 1; }

export DOCKER_IMAGE_NAME=vllm-disagg:glmv5.1-v0.24-local-batonfix
export IMAGE="$DOCKER_IMAGE_NAME"
export WANT_IMAGE_ID="${WANT_IMAGE_ID:-eb7d32be80bc}"
export IMAGE_TAR=/shared_inference/mdeopuja/model_blog_logs/docker_images/glmv5.1-v0.24-local-batonfix.tar
export GLM_SKIP_PATCHERS=1
export SKIP_RUNTIME_PATCH=1
export PROXY_TYPE=vllm_router
export VLLM_GCN_ARCH=gfx942
export AITER_BATON_TIMEOUT=300     # steal an abandoned aiter build lock after 5 min

export NIAH_COMBOS="8000/2048,20000/2048,96000/2048"   # worst sizes; 2048 out (retrieval determinism)
export NIAH_ENDPOINT=completions
export NIAH_SEEDS="0"
export NIAH_REPEAT=6   # same prompt x6 -> run-to-run variance (issue #47042 core claim)
export NIAH_TIMEOUT=5400
export NIAH_TOPO="MI300X 1P1D EP8 (disagg, /v1/completions) [DETERMINISM PROBE: same prompt x6]"
export LOG_WAIT_TIMEOUT_SECONDS=9000
export MODEL_LOCAL_DIR=/shared_inference/models_blog

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
  for p in 36473 36367 13345 8405 61005 61555 9711 30000; do fuser -k -n tcp "$p" >/dev/null 2>&1 || true; done
  echo "[$(hostname)] cleanup done: $(docker ps -q | wc -l) containers left"'

echo "=== preflight --health (job $SLURM_JOB_ID) ==="
bash glm5.1_notes/preflight_stage.sh --health || {
    echo "preflight failed: node missing image/model or RAS-faulted GPUs; aborting." >&2
    exit 1
}

echo "=== launching 1p1d NIAH pr47766-baked-batonfix /v1/completions (job $SLURM_JOB_ID) ==="
bash glm5.1_notes/run_one_config.sh 1 1 niah_determ_probe_1p1d niah
