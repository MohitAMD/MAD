#!/bin/bash
# =============================================================================
# 1P1D EP8 disagg GLM-5.1-FP8 -> bring up WideEP disagg serving on the
# FROM-SCRATCH image (glm5.1-fp8-disagg:mi300x-fromscratch-aiter017, built entirely from
# source per GLM5.1-FP8.disagg.MI300X.from-scratch.Dockerfile) and run the FULL
# cohere-accuracy-eval-suite (glm51_suite.yaml, profile=pre_release) against the
# router endpoint. Fresh orchestration (NOT copied from the 205803/804 NIAH jobs).
#
# Submit (leaf-pinned, 2 nodes in ONE leaf):
#   sbatch -p amd-rccl -N 2 --gres=gpu:8 --time=12:00:00 --requeue \
#     -w <leafNodeA>,<leafNodeB> \
#     --job-name=glm-1p1d-es-fromscratch --export=ALL \
#     glm5.1_notes/sbatch_1p1d_evalsuite_fromscratch_a017.sh
# =============================================================================
#SBATCH --job-name=glm_1p1d_es_fromscratch_a017
#SBATCH --partition=amd-rccl
#SBATCH --nodes=2
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

# ---- image / deployment env (FROM-SCRATCH image) ----------------------------
export DOCKER_IMAGE_NAME=glm5.1-fp8-disagg:mi300x-fromscratch-aiter017
export IMAGE="$DOCKER_IMAGE_NAME"
export WANT_IMAGE_ID="${WANT_IMAGE_ID:-837f39d260d8}"
export IMAGE_TAR=/shared_inference/mdeopuja/model_blog_logs/docker_images/glm5.1-fp8-disagg-mi300x-fromscratch-aiter017.tar
export GLM_SKIP_PATCHERS=1     # from-scratch image already bakes Patch A + Patch B
export SKIP_RUNTIME_PATCH=1
export PROXY_TYPE=vllm_router
export VLLM_GCN_ARCH=gfx942
export AITER_BATON_TIMEOUT=1800
export LOG_WAIT_TIMEOUT_SECONDS=9000
export MODEL_LOCAL_DIR=/shared_inference/models_blog
export MORIIO_DEFER_TIMEOUT="${MORIIO_DEFER_TIMEOUT:-600}"
export VLLM_MORIIO_DEFERRED_TIMEOUT_S="${VLLM_MORIIO_DEFERRED_TIMEOUT_S:-600}"
export VLLM_MORIIO_TRANSFER_TIMEOUT_S="${VLLM_MORIIO_TRANSFER_TIMEOUT_S:-600}"

# ---- eval config ------------------------------------------------------------
MODEL_PATH=/shared_inference/models_blog/GLM-5.1-FP8   # overwritten by discovery
ENDPOINT=http://127.0.0.1:30000/v1
EVAL_SUITE=/home/mdeopuja/cohere/cohere-accuracy-eval-suite
LOGDIR=/shared_inference/mdeopuja/model_blog_logs/${SLURM_JOB_ID}
ACCOUT="$LOGDIR/accsuite_out"
mkdir -p "$LOGDIR" "$ACCOUT"
# Serving-hold window. Must exceed the *total* runtime of the benchmarks this job
# runs, or the endpoint is torn down before the last (reliable-first) benchmark
# starts -- aime hit "Connection refused" this way on 210724 (10h window expired
# after a 5.4h lcb generation). Env-overridable; default raised to 24h so an
# isolated long-reasoning benchmark (aime serial @ disagg) completes.
export KEEPALIVE_MINS="${KEEPALIVE_MINS:-1400}"

echo "=== staging image on all nodes (job $SLURM_JOB_ID) ==="
srun --overlap --ntasks-per-node=1 bash -c '
  have=$(docker images -q "'"$IMAGE"'" 2>/dev/null | head -c12)
  if [ "$have" != "'"$WANT_IMAGE_ID"'" ]; then
    echo "[$(hostname)] have=[$have] want='"$WANT_IMAGE_ID"'; loading from tar..."
    docker load -i "'"$IMAGE_TAR"'" >/dev/null 2>&1 && echo "[$(hostname)] loaded=$(docker images -q "'"$IMAGE"'" | head -c12)" || echo "[$(hostname)] LOAD_FAILED"
  else echo "[$(hostname)] image OK ($have)"; fi'

echo "=== pre-launch cleanup (job $SLURM_JOB_ID) ==="
srun --overlap --ntasks-per-node=1 bash -c '
  ids=$(docker ps -aq --filter name=container_GLM-5.1-FP8 2>/dev/null); [ -n "$ids" ] && docker rm -f $ids >/dev/null 2>&1
  ids=$(docker ps -aq --filter name=glm_evalsuite 2>/dev/null); [ -n "$ids" ] && docker rm -f $ids >/dev/null 2>&1
  for p in 36473 36367 13345 8405 61005 61555 9711 30000; do fuser -k -n tcp "$p" >/dev/null 2>&1 || true; done
  docker run --rm -v /tmp:/tmp --entrypoint bash "'"$IMAGE"'" -c "rm -f /tmp/vllm_cache*/aiter_jit/build/lock_* /tmp/vllm_cache*/*/aiter_jit/build/lock_* /opt/vllm_cache/aiter_jit/build/lock_* 2>/dev/null; true" >/dev/null 2>&1 || true
  echo "[$(hostname)] cleanup done"'

echo "=== preflight --health (job $SLURM_JOB_ID) ==="
bash glm5.1_notes/preflight_stage.sh --health || { echo "preflight failed; aborting." >&2; exit 1; }

# ---- bring up 1P1D EP8 and HOLD it open (background) -------------------------
echo "=== launching 1P1D EP8 keepalive hold (job $SLURM_JOB_ID) ==="
bash glm5.1_notes/run_one_config.sh 1 1 evalsuite_1p1d_fromscratch_a017 keepalive &
KA_PID=$!
echo "keepalive launcher pid=$KA_PID"

teardown() {
  echo "=== teardown (job $SLURM_JOB_ID) ==="
  kill "$KA_PID" >/dev/null 2>&1 || true
  srun --overlap --ntasks-per-node=1 bash -c '
    for n in glm_evalsuite_1p1d_fromscratch_a017 container_GLM-5.1-FP8; do
      ids=$(docker ps -aq --filter name=$n 2>/dev/null); [ -n "$ids" ] && docker rm -f $ids >/dev/null 2>&1; done
    true' >/dev/null 2>&1 || true
}
trap teardown EXIT

# ---- discover served_model_name ---------------------------------------------
echo "=== discovering served_model_name (job $SLURM_JOB_ID) ==="
for i in $(seq 1 120); do
  _s=$(grep -h -oE "served_model_name=[^,]+" "$LOGDIR"/prefill_NODE0.log "$LOGDIR"/decode_NODE1.log 2>/dev/null | head -1 | cut -d= -f2)
  if [ -n "$_s" ]; then MODEL_PATH="$_s"; echo "discovered served_model_name=$MODEL_PATH"; break; fi
  if ! kill -0 "$KA_PID" 2>/dev/null; then echo "keepalive launcher exited early during discovery"; break; fi
  sleep 30
done

# ---- wait for the router to serve a real completion (cold bring-up ~40-60min) -
echo "=== waiting for :30000 readiness (job $SLURM_JOB_ID) ==="
ready=0
for i in $(seq 1 200); do
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
  tail -30 "$LOGDIR"/decode_NODE1.log 2>/dev/null; tail -30 "$LOGDIR"/prefill_NODE0.log 2>/dev/null
  exit 1
fi

# ---- run the FULL cohere-accuracy-eval-suite against the disagg endpoint ------
echo "=== running accuracy suite in container (served=$MODEL_PATH) (job $SLURM_JOB_ID) ==="
cp -f "$REPO/glm5.1_notes/run_accsuite_disagg_in_container.sh" "$ACCOUT/run_accsuite.sh"
chmod +x "$ACCOUT/run_accsuite.sh"
docker run --rm --name glm_evalsuite_1p1d_fromscratch_a017 --network host \
  --ipc host --shm-size 32g \
  -v /shared_inference:/shared_inference -v /mnt/m2m_nobackup:/mnt/m2m_nobackup \
  -v "$EVAL_SUITE":/eval \
  -e EVAL_ENDPOINT="$ENDPOINT" -e EVAL_API_KEY=EMPTY \
  -e OPENAI_KEY=EMPTY -e OPENAI_API_KEY=EMPTY \
  -e EVAL_JUDGE_MODEL="$MODEL_PATH" -e EVAL_JUDGE_ENDPOINT="$ENDPOINT" \
  -e NO_PROXY='*' -e no_proxy='*' -e HTTP_PROXY='' -e http_proxy='' -e HTTPS_PROXY='' -e https_proxy='' \
  -e LMEVAL_HF_HOME="$ACCOUT/hf_cache" -e LCB_HF_HOME="$ACCOUT/hf_cache" \
  -e EVALSCOPE_CACHE="$ACCOUT/evalscope_cache" -e MODELSCOPE_CACHE="$ACCOUT/modelscope_cache" \
  -e LCB_HOME="$ACCOUT/LiveCodeBench" \
  -e SERVED="$MODEL_PATH" -e ACCOUT="$ACCOUT" -e ONLY_BENCH="${ONLY_BENCH:-niah_single_2,mmlu_pro_50,gsm8k_100,gpqa_diamond_mini,livecodebench_mini,aime_2025_mini,aa_lcr_mini}" \
  -e AIME_NUM_CONCURRENT="${AIME_NUM_CONCURRENT:-1}" -e AIME_MAX_GEN_TOKS="${AIME_MAX_GEN_TOKS:-32768}" \
  -e HF_ALLOW_CODE_EVAL=1 -e TOKENIZERS_PARALLELISM=false \
  --entrypoint bash "$IMAGE" "$ACCOUT/run_accsuite.sh" 2>&1 | tee "$LOGDIR/acc_suite_1p1d.log"

echo "=== results ==="
[ -f "$ACCOUT/results.json" ] && cat "$ACCOUT/results.json" || echo "NO results.json"
echo "=== DONE 1P1D evalsuite (job $SLURM_JOB_ID); results in $LOGDIR ==="
