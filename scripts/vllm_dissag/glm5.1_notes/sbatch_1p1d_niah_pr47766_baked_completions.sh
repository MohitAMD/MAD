#!/bin/bash
# =============================================================================
# 1P1D disagg NIAH for GLM-5.1-FP8 on the BAKED PR#47766 image
# (vllm-disagg:glmv5.1-v0.24-pr47766-baked = pr47766 image + DSA/MoRIIO patchers
# baked IN-SOURCE at build), via raw /v1/completions. Runs GLM_SKIP_PATCHERS=1
# (fixes already in the image; NO runtime patching).
#
# Companion to sbatch_1p1d_niah_pr47766_completions.sh (that one is the non-baked
# image + GLM_SKIP_PATCHERS=0 runtime patchers). Same NIAH grid for A/B comparison.
#
# Submit:
#   sbatch -p amd-rccl -N 2 --gres=gpu:8 --time=12:00:00 --requeue \
#          --exclude=useocpm2m-097-119 \
#          --job-name=glm-1p1d-niah-pr47766-baked-cmpl --export=ALL \
#          glm5.1_notes/sbatch_1p1d_niah_pr47766_baked_completions.sh 2>&1 \
#     | tee output_sbatch_1p1d_niah_pr47766_baked_completions.txt
# =============================================================================
#SBATCH --job-name=glm_1p1d_niah_pr47766_baked_cmpl
#SBATCH --partition=amd-rccl
#SBATCH --nodes=2
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

# --- BAKED PR#47766 image (DSA/MoRIIO patchers already in-source) -------------
export DOCKER_IMAGE_NAME=vllm-disagg:glmv5.1-v0.24-pr47766-baked
export IMAGE="$DOCKER_IMAGE_NAME"
export WANT_IMAGE_ID="${WANT_IMAGE_ID:-8faa473fbeb7}"
export IMAGE_TAR=/shared_inference/mdeopuja/model_blog_logs/docker_images/glmv5.1-v0.24-pr47766-baked.tar
# Fixes are baked in-source -> DO NOT run runtime patchers.
export GLM_SKIP_PATCHERS=1
export SKIP_RUNTIME_PATCH=1
export PROXY_TYPE=vllm_router
export VLLM_GCN_ARCH=gfx942

# --- NIAH grid (raw /v1/completions) + 96k/32k (same as the non-baked A/B pair) -
export NIAH_COMBOS="2000/2048,8000/2048,20000/2048,35000/2048,96000/32000"
export NIAH_ENDPOINT=completions
export NIAH_SEEDS="0,1,2"
export NIAH_TIMEOUT=5400
export NIAH_TOPO="MI300X 1P1D EP8 (disagg, /v1/completions) [v0.24.0 + PR#47766 6-field + patchers BAKED, GLM_SKIP_PATCHERS=1]"
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

# --- preflight: root-container stale-lock clear + GPU/RAS health ---------------
echo "=== preflight --health (job $SLURM_JOB_ID) ==="
bash glm5.1_notes/preflight_stage.sh --health || {
    echo "preflight failed: node missing image/model or RAS-faulted GPUs; aborting." >&2
    exit 1
}

# --- launch 1P1D NIAH (/v1/completions), patchers OFF (baked) -----------------
echo "=== launching 1p1d NIAH pr47766-BAKED /v1/completions (job $SLURM_JOB_ID) ==="
bash glm5.1_notes/run_one_config.sh 1 1 niah_pr47766_baked_1p1d_cmpl niah
