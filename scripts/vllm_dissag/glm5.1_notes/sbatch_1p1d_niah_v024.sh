#!/bin/bash
# =============================================================================
# 1P/1D disaggregated (2 nodes) NIAH accuracy run on the v0.24-rebased GLM image
# (vllm-disagg:glmv5.1-v0.24-local) through the REAL vllm-router.
#
# Runs as a proper sbatch job so the allocation is stable (interactive salloc
# --no-shell allocations were getting PREEMPTED mid-bring-up). preflight --health
# runs the amdsmi ASIC probe and ABORTS if any node's GPUs are RAS-faulted (the
# silent boot-killer -- node 119 failed HIP init this way), so we never launch
# onto a bad node.
#
# Submit (exclude known-bad node 119):
#   sbatch -p amd-rccl -N 2 --gres=gpu:8 --time=12:00:00 \
#          --exclude=useocpm2m-097-119 \
#          --job-name=glm-1p1d-niah-v024 --export=ALL \
#          glm5.1_notes/sbatch_1p1d_niah_v024.sh 2>&1 \
#     | tee output_sbatch_1p1d_niah_v024.txt
# =============================================================================
#SBATCH --job-name=glm_1p1d_niah_v024
#SBATCH --partition=amd-rccl
#SBATCH --nodes=2                 # 1P + 1D = 2 nodes
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --spread-job
#SBATCH --time=12:00:00
#SBATCH --output=/shared_inference/mdeopuja/model_blog_logs/sbatch-%j.out
#SBATCH --error=/shared_inference/mdeopuja/model_blog_logs/sbatch-%j.err

set -u
REPO=/home/mdeopuja/cohere/MAD/scripts/vllm_dissag
cd "$REPO" || { echo "cannot cd to $REPO"; exit 1; }

# --- v0.24-rebased GLM image (router built-in, rocm.py GCN-arch fix baked) -----
export DOCKER_IMAGE_NAME=vllm-disagg:glmv5.1-v0.24-local
export IMAGE="$DOCKER_IMAGE_NAME"
export GLM_SKIP_PATCHERS=1 SKIP_RUNTIME_PATCH=1
export PROXY_TYPE=vllm_router
# arch resolver fallback (belt-and-suspenders with the baked rocm.py patch)
export VLLM_GCN_ARCH=gfx942

# --- NIAH grid: the long-context sizes that exposed the disagg accuracy bug ----
export NIAH_WORDS="8000,32000,100000"
export NIAH_MAXTOK=32
export LOG_WAIT_TIMEOUT_SECONDS=9000

# Use the NFS model in-place (1P1D bring-up tolerates it; the 705 GiB/node NVMe
# copy is skipped to save tens of minutes). preflight sees it as already present.
export MODEL_LOCAL_DIR=/shared_inference/models_blog
export IMAGE_TAR=/shared_inference/mdeopuja/model_blog_logs/docker_images/glmv5.1-v0.24-local.tar

# --- ensure the (local-only) image is on every allocated node -----------------
# Match by image ID, not tag: a stale same-tagged image from an earlier build
# (e.g. the routerless first build) would otherwise be silently reused and the
# proxy node would have no vllm-router. Force a load whenever the ID differs.
echo "=== staging image on all nodes (job $SLURM_JOB_ID) ==="
export WANT_IMAGE_ID="${WANT_IMAGE_ID:-2b5fd64bb098}"
srun --overlap --ntasks-per-node=1 bash -c '
  have=$(docker images -q "'"$IMAGE"'" 2>/dev/null | head -c12)
  if [ "$have" != "'"$WANT_IMAGE_ID"'" ]; then
    echo "[$(hostname)] have=[$have] want='"$WANT_IMAGE_ID"'; loading from tar..."
    docker load -i "'"$IMAGE_TAR"'" >/dev/null 2>&1 && echo "[$(hostname)] loaded=$(docker images -q "'"$IMAGE"'" | head -c12)" || echo "[$(hostname)] LOAD_FAILED"
  else
    echo "[$(hostname)] image OK ($have)"
  fi'

# --- preflight: GPU/RAS health probe + stale aiter-lock clear (aborts on bad) --
echo "=== preflight --health (job $SLURM_JOB_ID) ==="
bash glm5.1_notes/preflight_stage.sh --health || {
    echo "preflight failed: a node is missing image/model or has RAS-faulted GPUs; aborting." >&2
    exit 1
}

# --- launch 1P1D NIAH through the router --------------------------------------
echo "=== launching 1p1d NIAH v024 (job $SLURM_JOB_ID) ==="
bash glm5.1_notes/run_one_config.sh 1 1 niah_v024_sb niah
