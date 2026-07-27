#!/bin/bash
# =============================================================================
# DUAL-REPLICA 1P1D (4 nodes) -- STRONG-SCALING / FIXED-TOTAL-WORK variant.
#
# Same dual-replica 1P1D deployment and synchronized equal-concurrency driver as
# sbatch_1p1d_dual_replica.sh, but in SPLIT_TOTAL mode:
#
#   * each replica is held at the SAME max_concurrency = con (identical per-engine
#     operating point to the single-replica job 200584), AND
#   * the TOTAL input/output tokens per cell (replica A + replica B) MATCH the
#     single-replica 200584 cell -- i.e. the fixed 200584 workload is SPLIT 50/50
#     across the two replicas (each gets con*NUM_PROMPTS_FACTOR/2 prompts).
#
# This is the apples-to-apples strong-scaling comparison: the single box and the
# 4-node dual deployment clear the IDENTICAL token workload, so dual-aggregate
# throughput vs 200584 throughput is a true same-job speedup (and mirrors a
# production router splitting a fixed request stream 50/50).
#
# Grid is pinned to 200584's (con 64/128/256; 8000/1000, 4000/4000, 8000/4000)
# so every cell's aggregate token counts line up exactly with the baseline.
#
# Submit:
#   cd /home/mdeopuja/cohere/MAD/scripts/vllm_dissag
#   sbatch -p amd-rccl -N 4 --gres=gpu:8 --time=12:00:00 --requeue \
#          --job-name=glm-1p1d-dualsplit --export=ALL \
#          glm5.1_notes/sbatch_1p1d_dual_replica_splitload.sh
# =============================================================================
#SBATCH --job-name=glm_1p1d_dualsplit
#SBATCH --partition=amd-rccl
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
# Same-leaf placement (REQUIRED for disagg KV transfer): allocate within ONE leaf switch.
#SBATCH --switches=1
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/shared_inference/mdeopuja/model_blog_logs/sbatch-dualsplit-%j.out
#SBATCH --error=/shared_inference/mdeopuja/model_blog_logs/sbatch-dualsplit-%j.err

set -u
CORE=/home/mdeopuja/cohere/MAD/scripts/vllm_dissag/glm5.1_notes/sbatch_1p1d_dual_replica.sh

# --- strong-scaling / fixed-total-work knobs ---
export SPLIT_TOTAL=1

# Grid pinned to job 200584 so per-cell AGGREGATE token counts match the baseline.
export BENCHMARK_COMBINATIONS="${BENCHMARK_COMBINATIONS:-8000/1000 4000/4000 8000/4000}"
export BENCHMARK_CON="${BENCHMARK_CON:-64 128 256}"
export WARMUPS="${WARMUPS:-2}"
export NUM_PROMPTS_FACTOR="${NUM_PROMPTS_FACTOR:-4}"   # aggregate prompts = con*4 (== 200584); each replica gets con*2

# Single-replica baseline (job 200584 long-context perf log) -> parsed to CSV by
# the core script to produce the scaling_vs_single column.
export BASELINE_LOG="${BASELINE_LOG:-/shared_inference/mdeopuja/model_blog_logs/200584/benchmark_long_context_200584_20260709_025846_xP1_yD1_GLM-5.1-FP8_CONCURRENCY.log}"

# Reuse the (split-capable) core orchestrator; its #SBATCH lines are inert here,
# it inherits this job's SLURM allocation + the env exported above.
exec bash "$CORE"
