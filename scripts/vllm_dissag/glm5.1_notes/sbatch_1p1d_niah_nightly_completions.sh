#!/bin/bash
# =============================================================================
# Replicates the 202770 experiment (1P1D disagg NIAH for GLM-5.1-FP8 on the ROCm
# NIGHTLY v0.24 image) BUT targeting the raw /v1/completions endpoint instead of
# /v1/chat/completions -- i.e. NO chat template and NO reasoning parser. This
# isolates the long-context sparse-MLA DECODE path from the reasoning-parser
# machinery (matches the issue #47042 "/v1/completions (no reasoning parser)" note).
#
# Submit:
#   sbatch -p amd-rccl -N 2 --gres=gpu:8 --time=12:00:00 --requeue \
#          --exclude=useocpm2m-097-119 \
#          --job-name=glm-1p1d-niah-nightly-cmpl --export=ALL \
#          glm5.1_notes/sbatch_1p1d_niah_nightly_completions.sh 2>&1 \
#     | tee output_sbatch_1p1d_niah_nightly_completions.txt
# =============================================================================
#SBATCH --job-name=glm_1p1d_niah_nightly_cmpl
#SBATCH --partition=amd-rccl
#SBATCH --nodes=2                 # 1P + 1D
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
# Same-leaf placement (REQUIRED for disagg KV transfer): allocate within ONE leaf switch.
#SBATCH --switches=1
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/shared_inference/mdeopuja/model_blog_logs/sbatch-%j.out
#SBATCH --error=/shared_inference/mdeopuja/model_blog_logs/sbatch-%j.err

set -u
REPO=/home/mdeopuja/cohere/MAD/scripts/vllm_dissag
cd "$REPO" || { echo "cannot cd to $REPO"; exit 1; }

# --- v0.24-rebased GLM image on ROCm NIGHTLY base (same as 202770) ------------
export DOCKER_IMAGE_NAME=vllm-disagg:glmv5.1-v0.24-nightly
export IMAGE="$DOCKER_IMAGE_NAME"
export WANT_IMAGE_ID="${WANT_IMAGE_ID:-1709ab483bd5}"
export IMAGE_TAR=/shared_inference/mdeopuja/model_blog_logs/docker_images/glmv5.1-v0.24-nightly.tar
export GLM_SKIP_PATCHERS=1 SKIP_RUNTIME_PATCH=1
export PROXY_TYPE=vllm_router
export VLLM_GCN_ARCH=gfx942

# --- NIAH grid matching issue #47042, but via raw /v1/completions -------------
# ENDPOINT=completions -> no chat template, no reasoning parser (thinking is N/A).
# Per-case ISL/OSL: 2k..35k retrieval sizes (2048 out) + a 96k/32k shape.
export NIAH_COMBOS="2000/2048,8000/2048,20000/2048,35000/2048,96000/32000"
export NIAH_ENDPOINT=completions
export NIAH_SEEDS="0,1,2"
export NIAH_TOPO="MI300X 1P1D EP8 (disagg, /v1/completions no reasoning parser)"
export LOG_WAIT_TIMEOUT_SECONDS=9000
export MODEL_LOCAL_DIR=/shared_inference/models_blog

# --- ensure the correct image (by ID) is on every allocated node --------------
echo "=== staging image on all nodes (job $SLURM_JOB_ID) ==="
srun --overlap --ntasks-per-node=1 bash -c '
  have=$(docker images -q "'"$IMAGE"'" 2>/dev/null | head -c12)
  if [ "$have" != "'"$WANT_IMAGE_ID"'" ]; then
    echo "[$(hostname)] have=[$have] want='"$WANT_IMAGE_ID"'; loading from tar..."
    docker load -i "'"$IMAGE_TAR"'" >/dev/null 2>&1 && echo "[$(hostname)] loaded=$(docker images -q "'"$IMAGE"'" | head -c12)" || echo "[$(hostname)] LOAD_FAILED"
  else
    echo "[$(hostname)] image OK ($have)"
  fi'

# --- pre-launch cleanup of stale GLM containers/ports (avoid EADDRINUSE) -------
echo "=== pre-launch cleanup (job $SLURM_JOB_ID) ==="
srun --overlap --ntasks-per-node=1 bash -c '
  ids=$(docker ps -aq --filter name=container_GLM-5.1-FP8 2>/dev/null)
  [ -n "$ids" ] && docker rm -f $ids >/dev/null 2>&1
  for p in 36473 36367 13345 8405 61005 61555 9711 30000; do fuser -k -n tcp "$p" >/dev/null 2>&1 || true; done
  echo "[$(hostname)] cleanup done: $(docker ps -q | wc -l) containers left"'

# --- preflight: GPU/RAS health probe + stale aiter-lock clear -----------------
echo "=== preflight --health (job $SLURM_JOB_ID) ==="
bash glm5.1_notes/preflight_stage.sh --health || {
    echo "preflight failed: node missing image/model or RAS-faulted GPUs; aborting." >&2
    exit 1
}

# --- launch 1P1D NIAH (/v1/completions) through the router --------------------
echo "=== launching 1p1d NIAH nightly /v1/completions (job $SLURM_JOB_ID) ==="
bash glm5.1_notes/run_one_config.sh 1 1 niah_nightly_1p1d_cmpl niah
