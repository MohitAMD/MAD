#!/bin/bash
# =============================================================================
# Repeats the 202790 experiment (1P1D disagg NIAH, GLM-5.1-FP8, raw /v1/completions
# -- NO chat template, NO reasoning parser) but on the v0.24.0 RELEASE base image
# (vllm-disagg:glmv5.1-v0.24-local, built on vllm/vllm-openai-rocm:v0.24.0) instead
# of the ROCm nightly. Direct release-vs-nightly comparison at the raw-decode path.
#
# Submit:
#   sbatch -p amd-rccl -N 2 --gres=gpu:8 --time=12:00:00 --requeue \
#          --exclude=useocpm2m-097-119 \
#          --job-name=glm-1p1d-niah-v024rel-cmpl --export=ALL \
#          glm5.1_notes/sbatch_1p1d_niah_v024rel_completions.sh 2>&1 \
#     | tee output_sbatch_1p1d_niah_v024rel_completions.txt
# =============================================================================
#SBATCH --job-name=glm_1p1d_niah_v024rel_cmpl
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

# --- v0.24.0 RELEASE-base GLM image (vs 202790's nightly base) ----------------
export DOCKER_IMAGE_NAME=vllm-disagg:glmv5.1-v0.24-local
export IMAGE="$DOCKER_IMAGE_NAME"
export WANT_IMAGE_ID="${WANT_IMAGE_ID:-2b5fd64bb098}"
export IMAGE_TAR=/shared_inference/mdeopuja/model_blog_logs/docker_images/glmv5.1-v0.24-local.tar
export GLM_SKIP_PATCHERS=1 SKIP_RUNTIME_PATCH=1
export PROXY_TYPE=vllm_router
export VLLM_GCN_ARCH=gfx942

# --- NIAH grid matching #47042, via raw /v1/completions (no reasoning parser) --
# Per-case ISL/OSL (input words / output max_tokens): the 2k..35k retrieval sizes
# (2048 out) PLUS a 96k/32k ISL/OSL shape (96k-word context, 32k-token decode cap).
export NIAH_COMBOS="2000/2048,8000/2048,20000/2048,35000/2048,96000/32000"
export NIAH_ENDPOINT=completions
export NIAH_SEEDS="0,1,2"
export NIAH_TOPO="MI300X 1P1D EP8 (disagg, /v1/completions no reasoning parser) [v0.24.0 release base]"
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
echo "=== launching 1p1d NIAH v0.24.0-release /v1/completions (job $SLURM_JOB_ID) ==="
bash glm5.1_notes/run_one_config.sh 1 1 niah_v024rel_1p1d_cmpl niah
