#!/bin/bash
# =============================================================================
# dual_replica_sync_driver.sh
#
# Synchronized, EQUAL-CONCURRENCY load generator for a dual-replica 1P1D GLM-5.1
# deployment (two independent 1P1D EP8 disagg replicas, A and B).
#
# For every (ISL/OSL, concurrency) cell it fires the SAME `vllm bench serve`
# workload at BOTH replica routers at (near-)the same instant and barriers on
# both finishing, so both replicas are always held at the IDENTICAL max
# concurrency at the SAME time. This is what lets us test the "hold both
# replicas at cN simultaneously -> does aggregate throughput ~2x a single
# replica" question directly.
#
# Runs INSIDE the vLLM image (needs `vllm bench serve`) with `--network host`
# so it can reach each replica router by IP. Emits ONE
#   *_xP1_yD1_<MODEL>_CONCURRENCY.log
# per replica, in the exact shape benchmark_parser.py expects, then parses each
# into a per-replica CSV whose schema is IDENTICAL to job 200584's csv_results
# (Model,xP_yD,ISL,OSL,Concurrency,Prompts,...,Median_TPOT_ms).
#
# Required env:
#   A_HOST / B_HOST   replica-A / replica-B router IP (reachable from here)
#   OUTDIR            directory for logs + CSVs
#   REPO              path to the vllm_dissag repo (for benchmark_parser.py)
# Optional env (defaults chosen to MATCH job 200584 -- override to re-match):
#   A_PORT/B_PORT           router ports            (default 30000)
#   MODEL_PATH              served model path       (default GLM-5.1-FP8 blob)
#   MODEL_NAME             model tag in filenames  (default GLM-5.1-FP8)
#   BENCHMARK_CON          concurrency list        (default "8 16 32 64 128")
#   BENCHMARK_COMBINATIONS ISL/OSL list            (default "8000/1000 4000/4000 8000/4000")
#   WARMUPS                --num-warmups per cell   (default 2)
#   NUM_PROMPTS_FACTOR     prompts = factor*con     (default 4, min 16)
#   STEP_TIMEOUT           base per-cell timeout(s) (default 2400, scaled by tokens)
# =============================================================================
set -u

MODEL_PATH="${MODEL_PATH:-/mnt/m2m_nobackup/models_blog/GLM-5.1-FP8}"
MODEL_NAME="${MODEL_NAME:-GLM-5.1-FP8}"

A_HOST="${A_HOST:?set A_HOST to replica-A router IP}"
B_HOST="${B_HOST:?set B_HOST to replica-B router IP}"
A_PORT="${A_PORT:-30000}"
B_PORT="${B_PORT:-30000}"

OUTDIR="${OUTDIR:?set OUTDIR}"
REPO="${REPO:?set REPO to the vllm_dissag repo dir}"
mkdir -p "$OUTDIR"

# Never route benchmark/router traffic through a corp proxy.
export NO_PROXY='*' no_proxy='*' HTTP_PROXY='' http_proxy='' HTTPS_PROXY='' https_proxy=''

ts=$(date +%Y%m%d_%H%M%S)
# Filenames MUST carry _xP1_yD1_<MODEL>_CONCURRENCY so benchmark_parser.py can
# recover Model / xP_yD (it reads them from the filename, not the log body).
A_LOG="$OUTDIR/replicaA_${ts}_xP1_yD1_${MODEL_NAME}_CONCURRENCY.log"
B_LOG="$OUTDIR/replicaB_${ts}_xP1_yD1_${MODEL_NAME}_CONCURRENCY.log"

CON="${BENCHMARK_CON:-8 16 32 64 128}"
IFS=' ' read -ra COMBOS <<< "${BENCHMARK_COMBINATIONS:-8000/1000 4000/4000 8000/4000}"
WARMUPS="${WARMUPS:-2}"
NUM_PROMPTS_FACTOR="${NUM_PROMPTS_FACTOR:-4}"
STEP_TIMEOUT="${STEP_TIMEOUT:-2400}"

# Prompt-count mode:
#   SPLIT_TOTAL=0 (default, "per_replica"): each replica runs the FULL single-
#     replica cell (prompts = con*NUM_PROMPTS_FACTOR). Aggregate token volume is
#     2x the single-replica reference -> weak scaling.
#   SPLIT_TOTAL=1 ("split_total"): the AGGREGATE prompts across both replicas
#     equal the single-replica cell (con*NUM_PROMPTS_FACTOR); each replica gets
#     half. Per-replica max_concurrency is still `con`. Aggregate token volume
#     MATCHES the single-replica reference -> strong scaling / fixed problem size.
SPLIT_TOTAL="${SPLIT_TOTAL:-0}"
MODE=$([ "$SPLIT_TOTAL" = "1" ] && echo "split_total (aggregate == single-replica)" || echo "per_replica (aggregate == 2x single-replica)")

# Run one measured cell against one replica. The `[RUNNING] isl=.. osl=.. con=..
# warmups=.. prompts=..` header is the delimiter benchmark_parser.py splits on.
run_cell() {  # host port isl osl con nprompts timeout logfile
  local host=$1 port=$2 isl=$3 osl=$4 con=$5 np=$6 to=$7 log=$8
  echo "[RUNNING] isl=$isl osl=$osl con=$con warmups=$WARMUPS prompts=$np (timeout ${to}s)" >> "$log"
  timeout "$to" vllm bench serve \
    --model "$MODEL_PATH" \
    --backend vllm \
    --host "$host" \
    --port "$port" \
    --dataset-name random \
    --random-input-len "$isl" \
    --random-output-len "$osl" \
    --random-prefix-len 0 \
    --num-prompts "$np" \
    --num-warmups "$WARMUPS" \
    --request-rate inf \
    --ignore-eos \
    --max-concurrency "$con" >> "$log" 2>&1
  local rc=$?
  [ "$rc" -eq 124 ] && echo "[STALL] isl=$isl osl=$osl con=$con timed out after ${to}s" >> "$log"
}

echo "==== dual-replica SYNC driver ===="
echo "  A = ${A_HOST}:${A_PORT}"
echo "  B = ${B_HOST}:${B_PORT}"
echo "  combos = '${COMBOS[*]}'   con = '${CON}'   warmups=${WARMUPS} factor=${NUM_PROMPTS_FACTOR}"
echo "  prompt mode = ${MODE}"
echo "  A_LOG = $A_LOG"
echo "  B_LOG = $B_LOG"

for combo in "${COMBOS[@]}"; do
  IFS="/" read -r isl osl <<< "$combo"
  for con in $CON; do
    np_ref=$(( con * NUM_PROMPTS_FACTOR )); [ "$np_ref" -lt 16 ] && np_ref=16
    if [ "$SPLIT_TOTAL" = "1" ]; then
      # Aggregate (A+B) matches the single-replica cell; split 50/50 per replica.
      np=$(( np_ref / 2 )); [ "$np" -lt 1 ] && np=1
    else
      np=$np_ref
    fi
    tot=$(( isl + osl ))
    to=$(( STEP_TIMEOUT * tot / 2048 )); [ "$to" -lt "$STEP_TIMEOUT" ] && to=$STEP_TIMEOUT
    echo "=== cell isl=$isl osl=$osl con=$con prompts/replica=$np aggregate=$(( np * 2 )) single-ref=$np_ref -> firing BOTH replicas simultaneously (timeout ${to}s) $(date '+%H:%M:%S') ==="
    # Launch both replicas back-to-back (sub-ms skew vs a multi-minute cell),
    # then barrier so the NEXT cell only starts once BOTH replicas are idle.
    run_cell "$A_HOST" "$A_PORT" "$isl" "$osl" "$con" "$np" "$to" "$A_LOG" &
    pidA=$!
    run_cell "$B_HOST" "$B_PORT" "$isl" "$osl" "$con" "$np" "$to" "$B_LOG" &
    pidB=$!
    wait "$pidA"; wait "$pidB"
    sleep 10
  done
done

echo "=== dual-replica sweep complete; parsing per-replica CSVs ==="
# Per-replica CSVs in the SAME schema as job 200584's csv_results/*.csv.
python3 "$REPO/benchmark_parser.py" "$A_LOG" --csv "$OUTDIR/replicaA.csv" --no-screen \
  || echo "WARN: replicaA parse failed"
python3 "$REPO/benchmark_parser.py" "$B_LOG" --csv "$OUTDIR/replicaB.csv" --no-screen \
  || echo "WARN: replicaB parse failed"
echo "  -> $OUTDIR/replicaA.csv"
echo "  -> $OUTDIR/replicaB.csv"
echo "==== dual-replica SYNC driver done ===="
