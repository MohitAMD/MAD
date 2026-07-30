#!/bin/bash
# =============================================================================
# A/B test -- PEAK total tok/s per GPU on 4000/4000 ONLY, one topology per submit.
# Each arm runs with its OWN tuned params pushed to that deployment's safe max
# (gpu-util, max-num-batched-tokens, max-num-seqs) and sweeps concurrency to find
# its peak. FROM-SCRATCH image (af5d3ad886cc).
#
# Per-arm submit (set -N to match AB_XP+AB_YD nodes; export the tuned params):
#   1P1D (16 GPU): AB_XP=1 AB_YD=1 GPU_MEMORY_UTILIZATION=0.85 BENCHMARK_CON="256 512 1024" \
#     MAX_NUM_SEQS=256 MAX_NUM_BATCHED_TOKENS=16384 \
#     sbatch -N 2 -x useocpm2m-097-135,useocpm2m-097-119 --job-name=glm-ab-1p1d-4k4k \
#       --export=ALL glm5.1_notes/sbatch_abtest_4k4k_pergpu.sh
#   1P2D (24 GPU): AB_XP=1 AB_YD=2 GPU_MEMORY_UTILIZATION=0.85 BENCHMARK_CON="512 1024" ... -N 3 ...
#   2P2D (32 GPU): AB_XP=2 AB_YD=2 GPU_MEMORY_UTILIZATION=0.90 BENCHMARK_CON="512 1024 2048" ... -N 4 ...
# =============================================================================
#SBATCH --job-name=glm_ab_4k4k
#SBATCH --partition=amd-rccl
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --switches=1
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/shared_inference/mdeopuja/model_blog_logs/sbatch-%j.out
#SBATCH --error=/shared_inference/mdeopuja/model_blog_logs/sbatch-%j.err

set -u
REPO=/home/mdeopuja/cohere/MAD/scripts/vllm_dissag
cd "$REPO" || { echo "cannot cd to $REPO"; exit 1; }

AB_XP="${AB_XP:-1}"
AB_YD="${AB_YD:-1}"

# ---- image / deployment env (FROM-SCRATCH image) ----------------------------
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

# ---- A/B tuned params (each arm exports its own values at submit) ------------
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
export MORIIO_DEFER_TIMEOUT="${MORIIO_DEFER_TIMEOUT:-1800}"
export WARMUPS="${WARMUPS:-2}"
export NUM_PROMPTS_FACTOR="${NUM_PROMPTS_FACTOR:-4}"

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

# ---- 4k/4k A/B: sweep concurrency for peak total tok/s/GPU ------------------
echo "=== A/B ${AB_XP}P${AB_YD}D 4k/4k: util=${GPU_MEMORY_UTILIZATION} seqs=${MAX_NUM_SEQS} batched=${MAX_NUM_BATCHED_TOKENS} con='${BENCHMARK_CON:-512}' (job $SLURM_JOB_ID) ==="
export BENCHMARK_COMBINATIONS="4000/4000"
export BENCHMARK_CON="${BENCHMARK_CON:-512}"
bash glm5.1_notes/run_one_config.sh "$AB_XP" "$AB_YD" abtest_4k4k_${AB_XP}p${AB_YD}d long_context

echo "=== DONE A/B ${AB_XP}P${AB_YD}D 4k/4k (job $SLURM_JOB_ID); results in /shared_inference/mdeopuja/model_blog_logs/${SLURM_JOB_ID}/ ==="
