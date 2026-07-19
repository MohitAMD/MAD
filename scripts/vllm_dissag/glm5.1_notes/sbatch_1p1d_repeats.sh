#!/bin/bash
# =============================================================================
# 1P1D disagg GLM-5.1-FP8 -> hold the server open and run the FULL Cohere
# Accuracy Eval Suite + the NIAH smoke probe against it, to compare 1P1D
# (WideEP disagg) vs the single-node (no-WideEP) baseline in the results image.
#
# Same image as the single-node baseline (vllm-disagg:glmv5.1-v0.24-local-batonfix)
# so kernels/patches are identical; the ONLY variable is colocated-vs-disaggregated.
#
# Flow (all on node0 = prefill master + proxy, where vllm-router binds :30000):
#   1. stage image + preflight (2 nodes)
#   2. bring up 1P1D EP8 in the BACKGROUND via run_one_config.sh <keepalive>
#      (holds the server up KEEPALIVE_MINS with a light heartbeat)
#   3. poll :30000 until the router serves a real completion
#   4. NIAH smoke probe (benchmark_niah.py, /v1/completions, 2K/8K/20K/~96K)
#   5. cohere-eval-suite profile=image_match (NIAH, AA-LCR, LiveCodeBench,
#      MMLU-Pro, AIME 2025, GPQA Diamond) via /v1/chat/completions
#   6. collect results into the job log dir, tear down
#
# Submit:
#   sbatch -p amd-rccl -N 2 --gres=gpu:8 --time=12:00:00 --requeue \
#          --exclude=useocpm2m-097-119 \
#          --job-name=glm-1p1d-evalsuite --export=ALL \
#          glm5.1_notes/sbatch_1p1d_evalsuite.sh
# =============================================================================
#SBATCH --job-name=glm_1p1d_repeats
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

# ---- image / deployment env (matches the single-node baseline image) ---------
export DOCKER_IMAGE_NAME=vllm-disagg:glmv5.1-v0.24-local-batonfix
export IMAGE="$DOCKER_IMAGE_NAME"
export WANT_IMAGE_ID="${WANT_IMAGE_ID:-eb7d32be80bc}"
export IMAGE_TAR=/shared_inference/mdeopuja/model_blog_logs/docker_images/glmv5.1-v0.24-local-batonfix.tar
export GLM_SKIP_PATCHERS=1
export SKIP_RUNTIME_PATCH=1
export PROXY_TYPE=vllm_router
export VLLM_GCN_ARCH=gfx942
# Cold JIT cache + 8 EP ranks racing on aiter rmsnorm build -> a slow rank's lock
# gets stolen at 300s and its later baton.release() hits FileNotFoundError, killing
# decode engine init. Give cold builds much more headroom so ranks don't steal early.
export AITER_BATON_TIMEOUT=1800
export LOG_WAIT_TIMEOUT_SECONDS=9000
export MODEL_LOCAL_DIR=/shared_inference/models_blog

# ---- eval config -------------------------------------------------------------
MODEL_PATH=/mnt/m2m_nobackup/models_blog/GLM-5.1-FP8
ENDPOINT=http://127.0.0.1:30000/v1
EVAL_SUITE=/home/mdeopuja/cohere/cohere-eval-suite
LOGDIR=/shared_inference/mdeopuja/model_blog_logs/${SLURM_JOB_ID}
mkdir -p "$LOGDIR"
export KEEPALIVE_MINS=600     # hold the server up long enough for bring-up + full suite

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

echo "=== clean stale aiter JIT build locks on all nodes (job $SLURM_JOB_ID) ==="
# Locks are created by container-root under the node-local JIT cache; remove ONLY the
# lock_* files (keep already-built modules) so a fresh bring-up doesn't inherit a
# stale/half-held baton. Done as root via a throwaway container.
srun --overlap --ntasks-per-node=1 bash -c '
  docker run --rm -v /tmp:/tmp --entrypoint bash "'"$IMAGE"'" -c "
    rm -f /tmp/vllm_cache*/aiter_jit/build/lock_* /tmp/vllm_cache*/*/aiter_jit/build/lock_* 2>/dev/null
    rm -f /opt/vllm_cache/aiter_jit/build/lock_* 2>/dev/null
    true" >/dev/null 2>&1 || true
  echo "[$(hostname)] aiter JIT locks cleaned"'

echo "=== preflight --health (job $SLURM_JOB_ID) ==="
bash glm5.1_notes/preflight_stage.sh --health || {
    echo "preflight failed: node missing image/model or RAS-faulted GPUs; aborting." >&2
    exit 1
}

# ---- 2. bring up 1P1D and HOLD it open (background) ---------------------------
echo "=== launching 1P1D EP8 keepalive hold (job $SLURM_JOB_ID) ==="
bash glm5.1_notes/run_one_config.sh 1 1 evalsuite_1p1d keepalive &
KA_PID=$!
echo "keepalive launcher pid=$KA_PID"

teardown() {
  echo "=== teardown (job $SLURM_JOB_ID) ==="
  kill "$KA_PID" >/dev/null 2>&1 || true
  srun --overlap --ntasks-per-node=1 bash -c '
    ids=$(docker ps -aq --filter name=glm_evalsuite_1p1d 2>/dev/null)
    [ -n "$ids" ] && docker rm -f $ids >/dev/null 2>&1
    ids=$(docker ps -aq --filter name=container_GLM-5.1-FP8 2>/dev/null)
    [ -n "$ids" ] && docker rm -f $ids >/dev/null 2>&1
    true' >/dev/null 2>&1 || true
}
trap teardown EXIT

# ---- 3. wait for the router to serve a real completion (cold bring-up ~40min) -
echo "=== waiting for :30000 readiness (job $SLURM_JOB_ID) ==="
ready=0
# NOTE: max_tokens=1 requests HANG on this DSA/disagg build (the 1-token decode path
# stalls ~indefinitely); max_tokens>=3 return in ~1s. Use max_tokens=8 for the probe.
# Also bypass any proxy and allow a generous per-request timeout for the first
# (cold-shape) request. Poll on the proxy node's localhost (this batch script runs on
# the prefill-master/proxy node where vllm-router binds :30000).
for i in $(seq 1 200); do   # up to 200*30s = 100 min
  code=$(curl -s --noproxy '*' -o /tmp/ka_probe.$$ -w '%{http_code}' -m 120 \
    "http://127.0.0.1:30000/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL_PATH}\",\"prompt\":\"Ready?\",\"max_tokens\":8,\"temperature\":0}" 2>/dev/null)
  if [ "$code" = "200" ] && grep -q '"choices"' /tmp/ka_probe.$$ 2>/dev/null; then
    ready=1; echo "router ready after ~${i} probes (http $code)"; break
  fi
  if ! kill -0 "$KA_PID" 2>/dev/null; then echo "keepalive launcher exited early"; break; fi
  [ $((i % 4)) -eq 0 ] && echo "  ...still waiting (${i}) last_http=$code"
  sleep 30
done
if [ "$ready" != "1" ]; then
  echo "SERVER NOT READY -- tailing bring-up logs:" >&2
  tail -30 "$LOGDIR"/decode_NODE1.log 2>/dev/null
  tail -30 "$LOGDIR"/prefill_NODE0.log 2>/dev/null
  exit 1
fi

# ---- run AIME + GPQA repeats (concurrent) to average out disagg variance --------
REPEATS="${REPEATS:-3}"
CONCURRENCY="${CONCURRENCY:-8}"
echo "=== running AIME+GPQA repeats (x${REPEATS}, conc=${CONCURRENCY}) in container (job $SLURM_JOB_ID) ==="
docker run --rm --network host \
  -v "$HOME":"$HOME" \
  -v /shared_inference:/shared_inference \
  -v /mnt/m2m_nobackup:/mnt/m2m_nobackup \
  -e EVAL_ENDPOINT="$ENDPOINT" -e EVAL_API_KEY=EMPTY \
  -e NO_PROXY='*' -e no_proxy='*' -e HTTP_PROXY='' -e http_proxy='' -e HTTPS_PROXY='' -e https_proxy='' \
  -e MODEL_PATH="$MODEL_PATH" -e LOGDIR="$LOGDIR" -e EVAL_SUITE="$EVAL_SUITE" -e REPO="$REPO" \
  -e REPEATS="$REPEATS" -e CONCURRENCY="$CONCURRENCY" \
  --entrypoint bash "$IMAGE" -c '
    set -u
    python3 -c "import openai, yaml" 2>/dev/null || pip install --quiet "openai>=1.0.0" pyyaml >/dev/null 2>&1
    cd "$EVAL_SUITE"
    python3 runners/repeat_eval.py \
      --benchmark aime_2025_mini --benchmark gpqa_diamond_mini \
      --model "$MODEL_PATH" \
      --endpoint "http://127.0.0.1:30000/v1" \
      --repeats "$REPEATS" --concurrency "$CONCURRENCY" \
      --out "$LOGDIR/repeats_summary.json" 2>&1 | tee "$LOGDIR/repeats_1p1d.log"
    echo "repeats outputs -> $LOGDIR/repeats_summary.json"
  '

echo "=== DONE evals (job $SLURM_JOB_ID); results in $LOGDIR ==="
# teardown() runs on EXIT
