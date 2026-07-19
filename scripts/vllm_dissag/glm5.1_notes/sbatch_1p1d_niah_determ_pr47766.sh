#!/bin/bash
# =============================================================================
# 1P1D disagg NIAH determinism probe for GLM-5.1-FP8 on the MINIMAL-patch image
# (vllm-disagg:glmv5.1-v0.24-pr47766-baked-batonfix = stock v0.24.0 release +
# PR#47766 6-field metadata + baked DSA/MoRIIO patchers + aiter baton self-heal).
#
# PURPOSE: apples-to-apples disagg counterpart of the colocated TP=8 control
# (job 203697, image glmv5.1-v0.24-pr47766). Same 6-trial same-prompt probe,
# same fp8 KV, so the ONLY difference vs colocated is the disagg pipeline
# (EP8 mori all2all MoE + MoRIIO KV transfer/reindex). Isolates the Phase-0
# image confound from the earlier rebase+batonfix disagg run (203604, Δ=3@20k).
#
# Submit:
#   sbatch -p amd-rccl -N 2 --gres=gpu:8 --time=12:00:00 --requeue \
#          --job-name=glm-1p1d-determ-pr47766 --export=ALL \
#          glm5.1_notes/sbatch_1p1d_niah_determ_pr47766.sh
# =============================================================================
#SBATCH --job-name=glm_1p1d_determ_pr47766
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

export DOCKER_IMAGE_NAME=vllm-disagg:glmv5.1-v0.24-pr47766-baked-batonfix
export IMAGE="$DOCKER_IMAGE_NAME"
export WANT_IMAGE_ID="${WANT_IMAGE_ID:-aab075a8d1fe}"
export IMAGE_TAR=/shared_inference/mdeopuja/model_blog_logs/docker_images/glmv5.1-v0.24-pr47766-baked-batonfix.tar
export GLM_SKIP_PATCHERS=1
export SKIP_RUNTIME_PATCH=1
export PROXY_TYPE=vllm_router
export VLLM_GCN_ARCH=gfx942
export AITER_BATON_TIMEOUT=300     # steal an abandoned aiter build lock after 5 min

# fp8 KV -> matches colocated control (models.yaml default for GLM); apples-to-apples.
export KV_CACHE_DTYPE=fp8

export NIAH_COMBOS="8000/2048,20000/2048,96000/2048"   # worst sizes; 2048 out (retrieval determinism)
export NIAH_ENDPOINT=completions
export NIAH_SEEDS="0"
export NIAH_REPEAT=6   # same prompt x6 -> run-to-run variance (issue #47042 core claim)
export NIAH_TIMEOUT=5400
export NIAH_TOPO="MI300X 1P1D EP8 (disagg, /v1/completions) [DETERM PROBE x6 | v0.24.0+PR#47766 minimal | fp8 KV]"
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

echo "=== launching 1p1d NIAH determ pr47766-minimal /v1/completions (job $SLURM_JOB_ID) ==="
bash glm5.1_notes/run_one_config.sh 1 1 niah_determ_pr47766_1p1d niah
