#!/bin/bash
# Run ONE GLM config's accuracy (or accuracy_then_perf) directly as the salloc job step.
# Invoke from INSIDE the glm_res salloc shell:  bash ~/cohere_glmv5.1/run_one_config.sh <xP> <yD> <tag> [accuracy|accuracy_then_perf]
set -u
xP=$1; yD=$2; tag=$3; mode=${4:-accuracy}
REPO=/home/mdeopuja/cohere/MAD/scripts/vllm_dissag

cd "$REPO" || exit 1
# The recipe does `cd "$SLURM_SUBMIT_DIR"` (line 19) when run inside an salloc -> that
# points at where the salloc was created (~), not the script dir, so its REQUIRED_FILES
# check fails. Point SLURM_SUBMIT_DIR at the repo so the recipe cd's to the right place.
export SLURM_SUBMIT_DIR="$REPO"
export SLURM_OVERLAP=1                       # let the recipe's internal sruns share the step
export DOCKER_CONT_NAME="glm_${tag}_${SLURM_JOB_ID:-x}"   # unique per config (avoid name clash)
# clear any stale named container from a prior attempt on these nodes
docker rm -f "$DOCKER_CONT_NAME" >/dev/null 2>&1 || true

export DOCKER_IMAGE_NAME="${DOCKER_IMAGE_NAME:-rocm/pytorch-private:glm-dockerimage-built-09072026}"
export MODEL_NAME=GLM-5.1-FP8 RUN_MORI=1
# Unknown image provenance -> let the PR's idempotent/self-skipping DSA patchers run
# (they no-op if the image already carries the fixes in-source). Override to 1 for
# a known baked-fix image.
export GLM_SKIP_PATCHERS="${GLM_SKIP_PATCHERS:-0}"
export xP=$xP yD=$yD
export BENCHMARK_SCRIPT=$mode
# Accuracy gate uses the no-think NIAH harness (enable_thinking=false, small maxtok).
export NIAH_SCRIPT="${NIAH_SCRIPT:-niah_nothink.py}"
export NIAH_GATE_MIN="${NIAH_GATE_MIN:-8.0}"
export PREFER_SHM_MODEL=0          # NVMe model -> frees /dev/shm for vLLM IPC at high EP
export DOCKER_SHM_SIZE=128G
# NIAH disabled by default for now (harness needs fixing; sanity prompts are the gate).
# Set NIAH_CTX in the environment before calling to re-enable.
export NIAH_CTX="${NIAH_CTX-}"
export NIAH_DEPTHS="${NIAH_DEPTHS-}"
# perf grid (only used if mode=accuracy_then_perf)
export BENCHMARK_CON="${BENCHMARK_CON:-8 16 32 64}"
export BENCHMARK_COMBINATIONS="${BENCHMARK_COMBINATIONS:-1024/1024 4096/1024 8192/1024 56000/1024 100000/1024}"
export WARMUPS="${WARMUPS:-2}" NUM_PROMPTS_FACTOR="${NUM_PROMPTS_FACTOR:-4}"
export DECODE_CUDAGRAPH_MODE="${DECODE_CUDAGRAPH_MODE:-PIECEWISE}"  # branch/PR#176 default (FULL_DECODE_ONLY gave 0 TPOT benefit here)
# MoRI shmem heap is GPU-resident: 32GiB overflows HBM at EP<=16 (OOM). Scale by EP:
# 16GiB for EP<=16 (1P1D/2P2D), 32GiB for EP>=32 (4P4D/4P8D/8P4D).
_ep=$(( (xP>yD?xP:yD) * 8 ))
export MORI_SHMEM_HEAP_SIZE="${MORI_SHMEM_HEAP_SIZE:-17179869184}"  # default 16GiB; override per-run (e.g. 32GiB at EP32 w/ lower gpu-util)
# JIT cache MUST be node-local. /shared_inference is NFS; at EP>=32, 32 decode ranks
# cold-compiling Triton onto one NFS dir race -> "Errno 116 Stale file handle" mid
# cudagraph-capture (killed 4P/4D at 89%). Leave VLLM_CACHE_HOST_DIR UNSET so the
# recipe defaults to node-local /tmp/vllm_cache_<image-digest> (safe + still warm
# across reruns on the same pinned node).
export VLLM_CACHE_PERSIST=1
unset VLLM_CACHE_HOST_DIR

LOG=/home/mdeopuja/cohere/WideEP-GLM5.1/log_${mode}_${tag}.log
mkdir -p "$(dirname "$LOG")"

echo "=== run_one_config $tag xP=$xP yD=$yD mode=$mode $(date) ===" | tee "$LOG"
bash run_xPyD_models.slurm 2>&1 | tee -a "$LOG"
echo "=== run_one_config $tag DONE rc=${PIPESTATUS[0]} $(date) ===" | tee -a "$LOG"
