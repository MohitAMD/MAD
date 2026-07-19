#!/bin/bash
# =============================================================================
# 2P/2D disaggregated (4 nodes) perf run — ISL/OSL 96k/32k over con 64,128,256.
# Uses the sparse-MLA metadata fix (vllm #47766, auto-applied by the launcher).
# Submit with:
#   sbatch -p amd-rccl -N 4 --gres=gpu:8 --time=96:00:00 \
#          --job-name=glm-2p2d-96k --export=ALL \
#          glm5.1_notes/sbatch_2p2d_perf_96k_32k.sh 2>&1 \
#     | tee output_sbatch_2p2d_perf_96k_32k.txt
# =============================================================================
#SBATCH --job-name=glm_2p2d_perf96k_32k
#SBATCH --partition=amd-rccl
#SBATCH --nodes=4                 # 2P + 2D = 4 nodes
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --spread-job
#SBATCH --time=96:00:00
#SBATCH --output=/shared_inference/mdeopuja/model_blog_logs/sbatch-%j.out
#SBATCH --error=/shared_inference/mdeopuja/model_blog_logs/sbatch-%j.err

set -u
REPO=/home/mdeopuja/cohere/MAD/scripts/vllm_dissag
cd "$REPO" || { echo "cannot cd to $REPO"; exit 1; }

# --- image (single source of truth; preflight uses IMAGE, launcher uses DOCKER_IMAGE_NAME)
# Known-good prewarmed image (glm-dockerimage-built-09072026 had an aiter/LLVM JIT
# mismatch: clang rejects -amdgpu-coerce-illegal-types=1 -> prefill workers die).
export DOCKER_IMAGE_NAME=rocmshared/pytorch-private:glm5.1-47766-shiklatest_prewarmed_
export IMAGE="$DOCKER_IMAGE_NAME"
export GLM_SKIP_PATCHERS=1   # baked-fix image carries DSA fixes in-source

# --- perf knobs: ISL/OSL 96k/32k over the requested concurrencies ------------
export LOG_WAIT_TIMEOUT_SECONDS=9000
export BENCHMARK_CON="128 256 1024"
export BENCHMARK_COMBINATIONS="8000/1000 8000/4000 96000/32000"
# NOTE: DBO (--enable-dbo) is NOT compatible with the MoRI all2all backend
# (vLLM asserts microbatching supports only deepep_*/nixl_ep). Left OFF here.

# --- preflight: stage image + model + GPU health + clear stale aiter locks ----
echo "=== preflight (job $SLURM_JOB_ID) ==="
bash glm5.1_notes/preflight_stage.sh --health || {
    echo "preflight failed: not all nodes ready; aborting before launch." >&2
    exit 1
}

# --- run the disaggregated config (option A: perf-only, PR-native long_context) --
echo "=== launching 2p2d long_context 96k/32k (job $SLURM_JOB_ID) ==="
bash glm5.1_notes/run_one_config.sh 2 2 2p2d long_context
