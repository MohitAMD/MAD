#!/bin/bash
# =============================================================================
# Single-node, TP=8, colocated GLM-5.1-FP8 accuracy run using the UPDATED
# cohere-accuracy-eval-suite. Plain `vllm serve --tp 8` (NO disagg/MoRI/router).
# Image: vllm-disagg:glmv5.1-v0.25.1-pr47766-csfix-morishik (vLLM v0.25.1).
#
# Flow: stage image -> vllm serve (TP=8, --served-model-name glm-5-1-fp8) with
# --network host -> wait "Application startup complete" + curl /v1/models ->
# run the eval suite in a SECOND --network host container off the same image
# (repo mounted) -> tabulate -> teardown.
#
# Submit:
#   sbatch -p amd-rccl -N 1 --gres=gpu:8 --time=8:00:00 --requeue \
#          --exclude=useocpm2m-097-119 \
#          --job-name=glm-singlenode-tp8-accsuite --export=ALL \
#          glm5.1_notes/sbatch_singlenode_tp8_accsuite_morishik.sh
# =============================================================================
#SBATCH --job-name=glm-singlenode-tp8-accsuite
#SBATCH --partition=amd-rccl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/shared_inference/mdeopuja/model_blog_logs/sbatch-%j.out
#SBATCH --error=/shared_inference/mdeopuja/model_blog_logs/sbatch-%j.err

set -u
IMAGE=vllm-disagg:glmv5.1-v0.25.1-pr47766-csfix-morishik
WANT_ID=8ec10fdbf622
TAR=/shared_inference/mdeopuja/model_blog_logs/docker_images/glmv5.1-v0.25.1-pr47766-csfix-morishik.tar
MODEL=/mnt/m2m_nobackup/models_blog/GLM-5.1-FP8
[ -f "$MODEL/config.json" ] || MODEL=/shared_inference/models_blog/GLM-5.1-FP8
SERVED=glm-5-1-fp8
EVAL_REPO=/home/mdeopuja/cohere/cohere-accuracy-eval-suite
CTR="glm_sn_tp8_acc_${SLURM_JOB_ID}"
ECTR="glm_sn_tp8_eval_${SLURM_JOB_ID}"
LOGDIR="/shared_inference/mdeopuja/model_blog_logs/${SLURM_JOB_ID}"
ACCOUT="$LOGDIR/acc_suite_out"
JITCACHE="/mnt/m2m_nobackup/mdeopuja/vllm_jit_cache_sn_tp8"   # node-local, warm across retries
mkdir -p "$LOGDIR" "$ACCOUT"
echo "node=$(hostname) model=$MODEL image=$IMAGE served=$SERVED"

# --- stage image (by id) ------------------------------------------------------
have=$(docker images -q "$IMAGE" 2>/dev/null | head -c12)
if [ "$have" != "$WANT_ID" ]; then
  echo "loading image from tar..."; docker load -i "$TAR" >/dev/null 2>&1 && echo "loaded" || { echo "LOAD_FAILED"; exit 1; }
fi

# --- cleanup: stale containers + root-owned stale aiter locks -----------------
docker rm -f "$CTR" "$ECTR" >/dev/null 2>&1 || true
# Also kill ANY stray single-node serve/eval containers left by a prior/cancelled job
# on this node (scancel kills the SLURM step but detached docker containers keep
# holding GPU VRAM -> "Free memory < desired utilization" on the next job here).
_stray=$(docker ps -aq --filter name=glm_sn_tp8 2>/dev/null)
[ -n "$_stray" ] && docker rm -f $_stray >/dev/null 2>&1 || true
fuser -k -n tcp 30000 >/dev/null 2>&1 || true; sleep 3
mkdir -p "$JITCACHE"
docker run --rm -v /mnt/m2m_nobackup:/mnt/m2m_nobackup --entrypoint bash "$IMAGE" -c \
  'for L in /mnt/m2m_nobackup/*/vllm_jit_cache*/*/aiter_jit/build/lock_* /mnt/m2m_nobackup/*/vllm_jit_cache*/aiter_jit/build/lock_*; do rm -f "$L" 2>/dev/null; done; true' 2>/dev/null || true

# --- launch vllm serve (TP=8, colocated, host net) ----------------------------
docker run -d --name "$CTR" --network host \
  --device /dev/kfd --device /dev/dri --group-add video \
  --ipc host --shm-size 128g --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -v /shared_inference:/shared_inference -v /mnt/m2m_nobackup:/mnt/m2m_nobackup \
  -v "$JITCACHE":/opt/vllm_cache \
  -e VLLM_GCN_ARCH=gfx942 -e AITER_BATON_TIMEOUT=300 \
  -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_MOE=1 -e VLLM_ROCM_USE_AITER_MLA=1 \
  -e VLLM_ROCM_USE_AITER_RMSNORM=1 -e VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0 \
  -e VLLM_ROCM_USE_AITER_PAGED_ATTN=0 -e VLLM_USE_AITER_TRITON_SILU_MUL=0 \
  -e AITER_JIT_DIR=/opt/vllm_cache/aiter_jit -e VLLM_CACHE_ROOT=/opt/vllm_cache/vllm \
  -e TRITON_CACHE_DIR=/opt/vllm_cache/triton -e COMGR_CACHE_DIR=/opt/vllm_cache/comgr \
  --entrypoint bash "$IMAGE" -c "
    mkdir -p /opt/vllm_cache/aiter_jit /opt/vllm_cache/vllm /opt/vllm_cache/triton /opt/vllm_cache/comgr
    vllm serve '$MODEL' \
      --served-model-name '$SERVED' \
      --tensor-parallel-size 8 \
      --trust-remote-code \
      --kv-cache-dtype fp8 \
      --max-model-len 202752 \
      --gpu-memory-utilization 0.9 \
      --port 30000 2>&1 | tee $LOGDIR/serve.log
  "

# --- wait for server readiness (cold bring-up: weights + JIT, up to ~50 min) ---
echo "waiting for server readiness..."
ready=0
for i in $(seq 1 300); do
  if grep -q "Application startup complete" "$LOGDIR/serve.log" 2>/dev/null; then ready=1; break; fi
  if ! docker ps --format '{{.Names}}' | grep -q "^${CTR}$"; then echo "serve container exited early"; break; fi
  sleep 20
done
if [ "$ready" != "1" ]; then
  echo "SERVER NOT READY -- tail serve.log:"; tail -40 "$LOGDIR/serve.log" 2>/dev/null
  docker rm -f "$CTR" >/dev/null 2>&1 || true
  exit 1
fi
echo "server ready; sanity check /v1/models:"; curl -s -m 30 http://127.0.0.1:30000/v1/models 2>&1 | head -c 400; echo
echo "sanity /v1/completions:"; curl -s -m 60 http://127.0.0.1:30000/v1/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED\",\"prompt\":\"2+2=\",\"max_tokens\":5,\"temperature\":0}" 2>&1 | head -c 400; echo

# --- run the UPDATED eval suite in a second container off the same image -------
# Repo mounted rw at /eval; caches + outputs land under the shared job log dir.
# Use a standalone script (staged to the shared ACCOUT dir) to avoid inline docker
# -c quoting bugs (the prior run mangled '<18' as a redirection -> empty acc_suite.log).
mkdir -p "$ACCOUT"
cp -f /home/mdeopuja/cohere/MAD/scripts/vllm_dissag/glm5.1_notes/run_accsuite_in_container.sh "$ACCOUT/run_accsuite.sh"
chmod +x "$ACCOUT/run_accsuite.sh"
docker run --rm --name "$ECTR" --network host \
  --ipc host --shm-size 32g \
  -v /shared_inference:/shared_inference -v /mnt/m2m_nobackup:/mnt/m2m_nobackup \
  -v "$EVAL_REPO":/eval \
  -e EVAL_ENDPOINT=http://127.0.0.1:30000/v1 \
  -e EVAL_API_KEY=EMPTY \
  -e OPENAI_KEY=EMPTY -e OPENAI_API_KEY=EMPTY \
  -e EVAL_JUDGE_MODEL="$SERVED" \
  -e EVAL_JUDGE_ENDPOINT=http://127.0.0.1:30000/v1 \
  -e LMEVAL_HF_HOME="$ACCOUT/hf_cache" \
  -e LCB_HF_HOME="$ACCOUT/hf_cache" \
  -e EVALSCOPE_CACHE="$ACCOUT/evalscope_cache" \
  -e MODELSCOPE_CACHE="$ACCOUT/modelscope_cache" \
  -e LCB_HOME="$ACCOUT/LiveCodeBench" \
  -e SERVED="$SERVED" -e ACCOUT="$ACCOUT" \
  -e HF_ALLOW_CODE_EVAL=1 -e TOKENIZERS_PARALLELISM=false \
  --entrypoint bash "$IMAGE" "$ACCOUT/run_accsuite.sh" 2>&1 | tee "$LOGDIR/acc_suite.log"

echo "=== results ==="
[ -f "$ACCOUT/results.json" ] && cat "$ACCOUT/results.json" || echo "NO results.json"

echo "=== teardown ==="
docker rm -f "$CTR" "$ECTR" >/dev/null 2>&1 || true
echo "DONE_SN_ACCSUITE"
