#!/bin/bash
# =============================================================================
# RECIPE 17 + MTP — vLLM v0.27.1 + AITER 0.1.19 (bundled) + PR#176 WideEP overlay
# GLM-5.3-FP8 WideEP (EP8) 1P2D disagg — MTP (speculative decoding) ENABLED
#
# Goal: port Recipe 14's single-node MTP onto the Recipe 17 WideEP 1P2D stack.
# Recipe 17 already runs the exact stack Recipe 14 requires for correct MTP
# (vLLM v0.27.1 carries PR #47766/#45149/#47404/#48886, and decode runs
# FULL_AND_PIECEWISE cudagraphs). So enabling MTP is purely a matter of injecting
#   --speculative-config {"method":"mtp","num_speculative_tokens":N}
# into the per-role `vllm serve` argv via EXTRA_VLLM_ARGS (folded into
# MODEL_CONFIG_{PREFILL,DECODE} by vllm_disagg.sh, appended by connectors/moriio.sh).
#
# WHY BOTH ROLES: GLM-5.2/5.3 have num_nextn_predict_layers=1 (an MTP layer with
# its own KV). In P/D disagg the MoRIIO connector transfers KV blocks prefill->
# decode and is geometry-sensitive; prefill and decode MUST allocate the SAME
# layer/KV geometry. So spec-config is set GLOBALLY (EXTRA_VLLM_ARGS => both
# roles), not decode-only. Decode-only would give the decode engine an extra
# MTP-layer KV that prefill never produced -> transfer mismatch/hang.
#
# QUOTING: the JSON is wrapped in INNER single quotes so the connector's
# `eval "model_args=(${MODEL_CONFIG_DECODE})"` does NOT brace-expand {"a":..,"b":..}
# (verified: without inner quotes the JSON is shredded into 3 broken argv tokens).
#
# Toggle:  MTP=1 (default) NSPEC=3 ;  MTP=0 -> baseline (no speculative-config)
# Image: vllm-glm51-v027-aiter019-recipe15-wideep:prebaked4 (staged via tar)
#
# Submit (leaf-pinned, 2 nodes in ONE leaf):
#   sbatch -p amd-rccl -N 3 --gres=gpu:8 --time=8:00:00 --requeue --switches=1 \
#     --exclude=useocpm2m-097-[119,135,142] --job-name=r18_1p2d_glm53_mtp \
#     glm5.1_notes/sbatch_r18_1p2d_glm53_mtp.sh
# =============================================================================
#SBATCH --job-name=r18_1p2d_glm53_mtp
#SBATCH --partition=amd-rccl
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --switches=1
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/shared_inference/mdeopuja/model_blog_logs/sbatch-%j.out
#SBATCH --error=/shared_inference/mdeopuja/model_blog_logs/sbatch-%j.err

set -u
REPO=/home/mdeopuja/cohere/MAD/scripts/vllm_dissag
cd "$REPO" || { echo "cannot cd to $REPO"; exit 1; }

# ---- image -------------------------------------------------------------------
export DOCKER_IMAGE_NAME=vllm-glm51-v027-aiter019-recipe15-wideep:prebaked4
export IMAGE="$DOCKER_IMAGE_NAME"
IMAGE_TAR=/shared_inference/mdeopuja/model_blog_logs/docker_images/vllm-glm51-v027-aiter019-recipe15-wideep-prebaked4.tar

# Stage image on all nodes from tar (image is local-only, not in registry)
echo "=== staging image on all nodes (job ${SLURM_JOB_ID:-local}) ==="
srun --overlap --ntasks-per-node=1 bash -c "
  have=\$(docker images -q '${DOCKER_IMAGE_NAME}' 2>/dev/null | head -c12)
  if [ -z \"\$have\" ]; then
    echo \"[\$(hostname)] loading from tar (may take 3-4 min)...\"; docker load -i '${IMAGE_TAR}' 2>&1 | tail -2 && echo \"[\$(hostname)] LOAD_OK\" || { echo \"[\$(hostname)] LOAD_FAILED\"; exit 99; }
  else
    echo \"[\$(hostname)] image OK (\$have)\"
  fi
" || { echo "STAGING FAILED on one or more nodes — aborting job"; exit 1; }

# ---- recipe knobs (identical to sbatch_r17_1p1d_glm52_*.sh) -------------------
export MODEL_NAME=GLM-5.3-FP8
export RUN_MORI=1
export GLM_SKIP_PATCHERS=1
GLM52_FIX=/home/mdeopuja/cohere/MAD/scripts/vllm_dissag/glm5.1_notes/glm52_recipe15_fix.py
export EXTRA_DOCKER_ARGS="-v ${GLM52_FIX}:/usr/lib/python3.12/sitecustomize.py:ro -e TVM_FFI_DISABLE_TORCH_C_DLPACK=1"

export DECODE_CUDAGRAPH_MODE=FULL_AND_PIECEWISE   # required for the MTP TPOT win (UNIFORM_BATCH FULL graphs)
export PROXY_TYPE=vllm_router
export VLLM_GCN_ARCH=gfx942
export AITER_BATON_TIMEOUT=1800
export LOG_WAIT_TIMEOUT_SECONDS=9000
export MORIIO_DEFER_TIMEOUT=1800
export VLLM_CACHE_PERSIST=1
unset VLLM_CACHE_HOST_DIR 2>/dev/null || true
export AITER_JIT_DIR=/usr/local/lib/python3.12/dist-packages/aiter
# Recipe 14: prevent the engine self-killing during AITER JIT / first MTP forward.
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900

# ---- MTP (speculative decoding) injection -----------------------------------
MTP="${MTP:-1}"
NSPEC="${NSPEC:-3}"
if [ "${MTP}" = "1" ]; then
  # INNER single quotes around the JSON are REQUIRED (see header: brace-expansion).
  export EXTRA_VLLM_ARGS="--speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":${NSPEC}}'"
  echo "=== MTP ENABLED (n=${NSPEC}) on BOTH roles: EXTRA_VLLM_ARGS=${EXTRA_VLLM_ARGS} ==="
else
  export EXTRA_VLLM_ARGS=""
  echo "=== MTP DISABLED (baseline arm) ==="
fi

# ---- pre-launch cleanup on all nodes ----------------------------------------
echo "=== r17+mtp 1P2D: pre-launch cleanup (job ${SLURM_JOB_ID:-local}) ==="
srun --overlap --ntasks-per-node=1 bash -c '
  ids=$(docker ps -aq --filter name=container_${MODEL_NAME:-GLM} 2>/dev/null)
  [ -n "$ids" ] && docker rm -f $ids >/dev/null 2>&1 || true
  for p in 36473 36367 13345 8405 61005 61555 9711 30000 2222; do
    fuser -k -n tcp "$p" >/dev/null 2>&1 || true
  done
  echo "[$(hostname)] cleanup done"'

# ---- perf helper (long_context harness = per-shape warmup, steady-state) ----
run_perf() {
  local combos="$1" con="$2" tag="$3"
  echo "=== r17+mtp 1P2D PERF: combos='$combos' con='$con' tag='$tag' MTP=${MTP} (job ${SLURM_JOB_ID:-local}) ==="
  export BENCHMARK_COMBINATIONS="$combos"
  export BENCHMARK_CON="$con"
  bash glm5.1_notes/run_one_config.sh 1 2 "$tag" long_context
}

# ---- perf passes (MTP gain is largest at low concurrency; Recipe 14 peak @ MC=4) --
# Start with 8k/1k across the low-mid MC ladder to expose the MTP TPOT win, then
# an 8k/8k point (Recipe 14's headline shape).
run_perf "8000/1000"  "1 4 8 16 32"  r18_1p2d_glm53_mtp_8k1k
run_perf "8000/8000"  "1 4 8"        r18_1p2d_glm53_mtp_8k8k

# ---- post-run validation -----------------------------------------------------
JOB_LOG_DIR="/shared_inference/mdeopuja/model_blog_logs/${SLURM_JOB_ID:-local}"
DECODE_LOG="${JOB_LOG_DIR}/decode_NODE1.log"
echo ""
echo "############################################################################"
echo "# DONE  Recipe 17 + MTP (v0.27.1 + aiter019 + PR#176 overlay)  1P2D  MTP=${MTP} n=${NSPEC}"
echo "#   image : ${DOCKER_IMAGE_NAME}"
echo "#   logs  : ${JOB_LOG_DIR}/"
echo "# ---- MTP correctness gate (Recipe 14 token-0 '!' corruption check) ----"
# A corrupt MTP build emits runs of '!' (PLACEHOLDER_TOKEN_ID leak). The perf
# harness's warmup completion in the shape logs is the sample stream; scan the
# router/decode logs' sample completion for '!!!!' runs.
if grep -rqaE '!!!!!!' "${JOB_LOG_DIR}"/benchmark_long_context_*_CONCURRENCY.log "${JOB_LOG_DIR}"/vllm_router_NODE0.log 2>/dev/null; then
  echo "#   TOKEN-0 GATE: *** FAIL *** — runs of '!' detected (MTP metadata corruption; cf. Recipe 14 §8)"
else
  echo "#   TOKEN-0 GATE: PASS — no '!' corruption runs found"
fi
echo "# ---- MTP engagement / acceptance (from decode log) ----"
grep -aE "cudagraph_mode|FULL|UNIFORM_BATCH|Speculative|speculative|num_spec|acceptance|Draft acceptance|mtp" "${DECODE_LOG}" 2>/dev/null | tail -20
echo "############################################################################"
