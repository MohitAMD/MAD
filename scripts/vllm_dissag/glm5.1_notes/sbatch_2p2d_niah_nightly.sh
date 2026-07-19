#!/bin/bash
# =============================================================================
# 2P/2D disaggregated (4 nodes, EP16 per role) NIAH repro for GLM-5.1-FP8 on the
# v0.24-rebased GLM image built on the ROCm NIGHTLY base
# (vllm-disagg:glmv5.1-v0.24-nightly), through the REAL vllm-router.
#
# FAITHFUL to vllm issue #47042: thinking ON (NIAH_THINKING=1) + max_tokens 2048 +
# sizes 2000/8000/20000/35000.
#
# Submit (exclude known-bad node 119):
#   sbatch -p amd-rccl -N 4 --gres=gpu:8 --time=12:00:00 --requeue \
#          --exclude=useocpm2m-097-119 \
#          --job-name=glm-2p2d-niah-nightly --export=ALL \
#          glm5.1_notes/sbatch_2p2d_niah_nightly.sh 2>&1 \
#     | tee output_sbatch_2p2d_niah_nightly.txt
# =============================================================================
#SBATCH --job-name=glm_2p2d_niah_nightly
#SBATCH --partition=amd-rccl
#SBATCH --nodes=4                 # 2P + 2D (EP16 per sub-cluster)
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

# --- v0.24-rebased GLM image on ROCm NIGHTLY base -----------------------------
export DOCKER_IMAGE_NAME=vllm-disagg:glmv5.1-v0.24-nightly
export IMAGE="$DOCKER_IMAGE_NAME"
export WANT_IMAGE_ID="${WANT_IMAGE_ID:-1709ab483bd5}"
export IMAGE_TAR=/shared_inference/mdeopuja/model_blog_logs/docker_images/glmv5.1-v0.24-nightly.tar
export GLM_SKIP_PATCHERS=1 SKIP_RUNTIME_PATCH=1
export PROXY_TYPE=vllm_router      # 2P2D needs the router's DP-rank round-robin
export VLLM_GCN_ARCH=gfx942
# Skip the boot memory-profiling forward: at cross-node EP16 that forward hits
# AITER's eager fp8-quant torch_guard bug -> decode EngineCore "RuntimeError:
# cancelled" in determine_available_memory (killed 202771/202669). Setting an
# explicit KV-cache budget makes vLLM skip profiling and allocate directly.
export KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-42949672960}"   # 40 GiB/rank
# "enforce eager" (SAFE form): cudagraph_mode=NONE for BOTH roles while KEEPING the
# +quant_fp8 custom op. The recipe forbids bare --enforce-eager on these AITER images
# (it routes fp8 quant through a signature-mismatched op). Removing decode's cudagraph
# capture cuts the number of aiter-JIT passes -> fewer chances for the module_moe_fmoe_asm
# baton deadlock that hung 202799 (prefill was already NONE and still deadlocked, so this
# reduces -- not guarantees -- the hang; a fresh run also re-rolls the baton race).
export PREFILL_CUDAGRAPH_MODE=NONE
export DECODE_CUDAGRAPH_MODE=NONE

# --- NIAH grid matching issue #47042 (thinking ON, 2048 tok, 2k..35k) ---------
export NIAH_WORDS="2000,8000,20000,35000"
export NIAH_THINKING=1
export NIAH_MAXTOK=2048
export NIAH_SEEDS="0,1,2"
export NIAH_TOPO="MI300X 2P2D EP16 (disagg, MoRIIO)"
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

# --- launch 2P2D NIAH through the router --------------------------------------
echo "=== launching 2p2d NIAH nightly (job $SLURM_JOB_ID) ==="
bash glm5.1_notes/run_one_config.sh 2 2 niah_nightly_2p2d niah
