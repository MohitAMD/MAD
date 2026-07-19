#!/bin/bash
# =============================================================================
# 2P/2D disaggregated (4 nodes, EP16 per role) PERF sweep on the v0.24-rebased
# GLM image (vllm-disagg:glmv5.1-v0.24-local) through the REAL vllm-router.
#
#   Concurrency      : 128, 256, 512, 1024
#   ISL/OSL combos   : 8k/1k, 8k/4k, 96k/32k
#
# Same robustness scaffolding as the validated 1P1D NIAH wrapper:
#   - image staged by IMAGE ID (not tag) so a stale same-tagged build can't be
#     silently reused (that left a proxy node with no vllm-router before).
#   - preflight --health aborts if any allocated node has RAS-faulted GPUs.
#   - stable sbatch job (interactive salloc allocations were getting preempted).
#
# Submit (exclude known-bad node 119):
#   sbatch -p amd-rccl -N 4 --gres=gpu:8 --time=48:00:00 \
#          --exclude=useocpm2m-097-119 \
#          --job-name=glm-2p2d-ep16-perf-v024 --export=ALL \
#          glm5.1_notes/sbatch_2p2d_ep16_perf_v024.sh 2>&1 \
#     | tee output_sbatch_2p2d_ep16_perf_v024.txt
# =============================================================================
#SBATCH --job-name=glm_2p2d_ep16_perf_v024
#SBATCH --partition=amd-rccl
#SBATCH --nodes=4                 # 2P + 2D = 4 nodes (EP16 per sub-cluster)
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --spread-job
#SBATCH --time=48:00:00
#SBATCH --requeue                 # auto-retry if preempted mid-bring-up (normal QOS)
#SBATCH --open-mode=append        # keep prior attempt's output on requeue
#SBATCH --output=/shared_inference/mdeopuja/model_blog_logs/sbatch-%j.out
#SBATCH --error=/shared_inference/mdeopuja/model_blog_logs/sbatch-%j.err

set -u
REPO=/home/mdeopuja/cohere/MAD/scripts/vllm_dissag
cd "$REPO" || { echo "cannot cd to $REPO"; exit 1; }

# --- v0.24-rebased GLM image (router built-in, rocm.py GCN-arch fix baked) -----
export DOCKER_IMAGE_NAME=vllm-disagg:glmv5.1-v0.24-local
export IMAGE="$DOCKER_IMAGE_NAME"
export GLM_SKIP_PATCHERS=1 SKIP_RUNTIME_PATCH=1
export PROXY_TYPE=vllm_router      # 2P2D needs the router's DP-rank round-robin
export VLLM_GCN_ARCH=gfx942

# --- perf grid: concurrency 128/256/512/1024 x ISL/OSL 8k/1k, 8k/4k, 96k/32k --
export LOG_WAIT_TIMEOUT_SECONDS=9000
export BENCHMARK_CON="128 256 512 1024"
export BENCHMARK_COMBINATIONS="8000/1000 8000/4000 96000/32000"
# NOTE: DBO (--enable-dbo) is NOT compatible with the MoRI all2all backend
# (vLLM asserts microbatching supports only deepep_*/nixl_ep). Left OFF here.

# NVMe model if already present; skip the 705 GiB/node copy otherwise (perf is
# unaffected once weights are in HBM). preflight sees the NFS copy as present.
export MODEL_LOCAL_DIR=/shared_inference/models_blog
export IMAGE_TAR=/shared_inference/mdeopuja/model_blog_logs/docker_images/glmv5.1-v0.24-local.tar

# --- ensure the correct image (by ID) is on every allocated node --------------
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

# --- clear stale GLM containers/processes holding rendezvous ports ------------
# A prior (preempted) run can leave a zombie container/process bound to the DP
# rendezvous TCPStore port (36473) -> the new decode master fails to bind with
# EADDRINUSE (-98) and the whole decode group hangs. Nodes are exclusive to this
# job, so removing our own container_GLM-* leftovers here is safe.
echo "=== pre-launch cleanup of stale GLM containers/ports (job $SLURM_JOB_ID) ==="
srun --overlap --ntasks-per-node=1 bash -c '
  ids=$(docker ps -aq --filter name=container_GLM-5.1-FP8 2>/dev/null)
  [ -n "$ids" ] && docker rm -f $ids >/dev/null 2>&1
  # kill any bare process still holding the rendezvous / rpc ports
  for p in 36473 36367 13345 8405 61005 61555 9711 30000; do
    fuser -k -n tcp "$p" >/dev/null 2>&1 || true
  done
  echo "[$(hostname)] cleanup done: containers=$(docker ps -q | wc -l) leftover"'

# --- preflight: GPU/RAS health probe + stale aiter-lock clear (aborts on bad) --
echo "=== preflight --health (job $SLURM_JOB_ID) ==="
bash glm5.1_notes/preflight_stage.sh --health || {
    echo "preflight failed: a node is missing image/model or has RAS-faulted GPUs; aborting." >&2
    exit 1
}

# --- launch 2P2D (EP16) perf sweep through the router -------------------------
echo "=== launching 2p2d EP16 long_context perf (job $SLURM_JOB_ID) ==="
bash glm5.1_notes/run_one_config.sh 2 2 ep16_perf_v024 long_context
