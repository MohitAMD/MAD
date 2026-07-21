#!/bin/bash
# =============================================================================
# DUAL-REPLICA 1P1D (4 nodes) GLM-5.1-FP8 -- synchronized equal-concurrency sweep
#
# Stands up TWO independent 1P1D EP8 disagg replicas from ONE 4-node allocation:
#   Replica A : nodes[0]=prefill+router(:30000)  nodes[1]=decode
#   Replica B : nodes[2]=prefill+router(:30000)  nodes[3]=decode
# Each replica is byte-for-byte the job-200584 1P1D recipe (same image, MoRI-EP,
# PROXY_TYPE=vllm_router), brought up via run_one_config.sh in `keepalive` mode
# and HELD open. A single external driver (dual_replica_sync_driver.sh) then
# drives BOTH routers at the IDENTICAL max concurrency at the SAME time, and the
# per-replica CSVs come out in the SAME schema as 200584's csv_results so the
# datapoints are directly comparable.
#
# Why two independent replicas (not one stretched router): KV transfer stays
# intra-replica (P_A->D_A, P_B->D_B). The router layer's only job is to keep both
# replicas pinned at the same concurrency, which we enforce from the driver.
#
# Submit:
#   cd /home/mdeopuja/cohere/MAD/scripts/vllm_dissag
#   sbatch -p amd-rccl -N 4 --gres=gpu:8 --time=12:00:00 --requeue \
#          --job-name=glm-1p1d-dual --export=ALL \
#          glm5.1_notes/sbatch_1p1d_dual_replica.sh
# =============================================================================
#SBATCH --job-name=glm_1p1d_dual_replica
#SBATCH --partition=amd-rccl
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --spread-job
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/shared_inference/mdeopuja/model_blog_logs/sbatch-dual-%j.out
#SBATCH --error=/shared_inference/mdeopuja/model_blog_logs/sbatch-dual-%j.err

set -u
REPO=/home/mdeopuja/cohere/MAD/scripts/vllm_dissag
cd "$REPO" || { echo "cannot cd to $REPO"; exit 1; }

# ---- image / deployment env (MUST match the single-replica 200584 baseline) --
export DOCKER_IMAGE_NAME="${DOCKER_IMAGE_NAME:-vllm-disagg:glmv5.1-v0.24-local-batonfix}"
export IMAGE="$DOCKER_IMAGE_NAME"
export WANT_IMAGE_ID="${WANT_IMAGE_ID:-eb7d32be80bc}"
export IMAGE_TAR="${IMAGE_TAR:-/shared_inference/mdeopuja/model_blog_logs/docker_images/glmv5.1-v0.24-local-batonfix.tar}"
export GLM_SKIP_PATCHERS=1
export SKIP_RUNTIME_PATCH=1
export PROXY_TYPE=vllm_router
export VLLM_GCN_ARCH=gfx942
export AITER_BATON_TIMEOUT=1800
export LOG_WAIT_TIMEOUT_SECONDS=9000
export MODEL_LOCAL_DIR=/shared_inference/models_blog
export MODEL_NAME=GLM-5.1-FP8
export RUN_MORI=1

# ---- sweep grid: keep IDENTICAL to job 200584 for direct comparability --------
export BENCHMARK_COMBINATIONS="${BENCHMARK_COMBINATIONS:-8000/1000 4000/4000 8000/4000}"
export BENCHMARK_CON="${BENCHMARK_CON:-8 16 32 64 128}"
export WARMUPS="${WARMUPS:-2}"
export NUM_PROMPTS_FACTOR="${NUM_PROMPTS_FACTOR:-4}"

# Hold each replica open long enough for cold bring-up (~40min) + the full sweep.
export KEEPALIVE_MINS="${KEEPALIVE_MINS:-600}"

MODEL_PATH=/mnt/m2m_nobackup/models_blog/GLM-5.1-FP8
ROUTER_PORT=30000
LOGROOT=/shared_inference/mdeopuja/model_blog_logs/dual_${SLURM_JOB_ID}
OUTDIR="$LOGROOT/driver"
mkdir -p "$OUTDIR"

# ---- 0. resolve the 4 nodes + IPs, split into replica A / B ------------------
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [ "${#NODES[@]}" -lt 4 ]; then
  echo "FATAL: need 4 nodes, got ${#NODES[@]} (${NODES[*]})" >&2; exit 1
fi
node_ip() { srun --overlap --nodes=1 --ntasks=1 --nodelist="$1" bash -c 'hostname -I' | awk '{print $1}'; }
A_NODES="${NODES[0]},${NODES[1]}"
B_NODES="${NODES[2]},${NODES[3]}"
A_ROUTER_IP="$(node_ip "${NODES[0]}")"   # replica A prefill-master + router
B_ROUTER_IP="$(node_ip "${NODES[2]}")"   # replica B prefill-master + router
echo "=== dual-replica layout (job $SLURM_JOB_ID) ==="
echo "  Replica A: $A_NODES  router http://${A_ROUTER_IP}:${ROUTER_PORT}"
echo "  Replica B: $B_NODES  router http://${B_ROUTER_IP}:${ROUTER_PORT}"

# ---- 1. stage image + cleanup across ALL 4 nodes -----------------------------
echo "=== staging image on all nodes ==="
srun --overlap --ntasks-per-node=1 bash -c '
  have=$(docker images -q "'"$IMAGE"'" 2>/dev/null | head -c12)
  if [ "$have" != "'"$WANT_IMAGE_ID"'" ]; then
    echo "[$(hostname)] loading image from tar..."
    docker load -i "'"$IMAGE_TAR"'" >/dev/null 2>&1 && echo "[$(hostname)] loaded" || echo "[$(hostname)] LOAD_FAILED"
  else
    echo "[$(hostname)] image OK ($have)"
  fi'

echo "=== pre-launch cleanup on all nodes ==="
srun --overlap --ntasks-per-node=1 bash -c '
  ids=$(docker ps -aq --filter name=container_GLM-5.1-FP8 2>/dev/null); [ -n "$ids" ] && docker rm -f $ids >/dev/null 2>&1
  ids2=$(docker ps -aq --filter name=glm_dual 2>/dev/null); [ -n "$ids2" ] && docker rm -f $ids2 >/dev/null 2>&1
  for p in 36473 36367 13345 8405 61005 61555 9711 30000 20005; do fuser -k -n tcp "$p" >/dev/null 2>&1 || true; done
  echo "[$(hostname)] cleanup done"'

# ---- 2. bring up BOTH replicas (keepalive), each pinned to its own 2 nodes ----
# Each replica is a full run_one_config.sh 1 1 (== 200584 recipe). We pin it to a
# 2-node subset by overriding SLURM_JOB_NODELIST, and give it its own LOG_PATH so
# the two recipe instances (same SLURM_JOB_ID) don't clobber each other's logs.
launch_replica() {  # tag nodelist logpath
  local tag=$1 nodelist=$2 logpath=$3
  mkdir -p "$logpath"
  (
    export SLURM_JOB_NODELIST="$nodelist"
    export SLURM_NODELIST="$nodelist"
    # Pin the node-COUNT vars to this replica's 2-node subset too. The recipe's
    # model-availability check runs `srun --nodes=$SLURM_NNODES ...` BEFORE it
    # recomputes SLURM_NNODES, so leaving the inherited 4 here makes srun ask for
    # 4 nodes out of a 2-node subset ("Only allocated 2 nodes asked for 4").
    export SLURM_NNODES=2 SLURM_NTASKS=2 SLURM_JOB_NUM_NODES=2 SLURM_NPROCS=2
    export SLURM_TASKS_PER_NODE="1(x2)"
    export LOG_PATH="$logpath"
    export ROUTER_PORT BENCHMARK_PORT="$ROUTER_PORT"
    bash glm5.1_notes/run_one_config.sh 1 1 "$tag" keepalive
  ) > "$logpath/bringup.log" 2>&1 &
  echo $!
}

echo "=== launching replica A (keepalive) ==="
KA_A=$(launch_replica dual_A "$A_NODES" "$LOGROOT/replicaA")
echo "  replica A launcher pid=$KA_A"
echo "=== launching replica B (keepalive) ==="
KA_B=$(launch_replica dual_B "$B_NODES" "$LOGROOT/replicaB")
echo "  replica B launcher pid=$KA_B"

teardown() {
  echo "=== teardown (job $SLURM_JOB_ID) ==="
  kill "$KA_A" "$KA_B" >/dev/null 2>&1 || true
  srun --overlap --ntasks-per-node=1 bash -c '
    ids=$(docker ps -aq --filter name=container_GLM-5.1-FP8 2>/dev/null); [ -n "$ids" ] && docker rm -f $ids >/dev/null 2>&1
    ids2=$(docker ps -aq --filter name=glm_dual 2>/dev/null); [ -n "$ids2" ] && docker rm -f $ids2 >/dev/null 2>&1
    true' >/dev/null 2>&1 || true
}
trap teardown EXIT

# ---- 3. wait until BOTH routers serve a real completion -----------------------
# (max_tokens=1 can hang on this DSA/disagg build; use 8 for the probe.)
wait_router() {  # ip
  local ip=$1 code
  for i in $(seq 1 200); do   # up to 200*30s = 100 min cold bring-up
    code=$(curl -s --noproxy '*' -o /dev/null -w '%{http_code}' -m 120 \
      "http://${ip}:${ROUTER_PORT}/v1/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"${MODEL_PATH}\",\"prompt\":\"Ready?\",\"max_tokens\":8,\"temperature\":0}" 2>/dev/null)
    [ "$code" = "200" ] && { echo "  router ${ip} ready after ~${i} probes"; return 0; }
    if ! kill -0 "$KA_A" 2>/dev/null || ! kill -0 "$KA_B" 2>/dev/null; then
      echo "  a replica launcher exited early while waiting on ${ip}"; return 1
    fi
    [ $((i % 4)) -eq 0 ] && echo "  ...waiting on router ${ip} (${i}) last_http=$code"
    sleep 30
  done
  return 1
}
echo "=== waiting for both routers to be ready ==="
if ! wait_router "$A_ROUTER_IP" || ! wait_router "$B_ROUTER_IP"; then
  echo "SERVER(S) NOT READY -- tailing bring-up logs:" >&2
  tail -40 "$LOGROOT"/replicaA/*/decode_NODE1.log 2>/dev/null
  tail -40 "$LOGROOT"/replicaB/*/decode_NODE1.log 2>/dev/null
  exit 1
fi
echo "=== BOTH replicas ready; starting synchronized equal-concurrency sweep ==="

# ---- 4. synchronized equal-concurrency driver (throwaway container) -----------
# --network host so it reaches both replica router IPs; same image => has vllm.
docker run --rm --network host \
  -v "$HOME":"$HOME" \
  -v /shared_inference:/shared_inference \
  -v /mnt/m2m_nobackup:/mnt/m2m_nobackup \
  -v "$REPO":"$REPO" \
  -e A_HOST="$A_ROUTER_IP" -e B_HOST="$B_ROUTER_IP" \
  -e A_PORT="$ROUTER_PORT" -e B_PORT="$ROUTER_PORT" \
  -e MODEL_PATH="$MODEL_PATH" -e MODEL_NAME="$MODEL_NAME" \
  -e OUTDIR="$OUTDIR" -e REPO="$REPO" \
  -e BENCHMARK_COMBINATIONS="$BENCHMARK_COMBINATIONS" \
  -e BENCHMARK_CON="$BENCHMARK_CON" \
  -e WARMUPS="$WARMUPS" -e NUM_PROMPTS_FACTOR="$NUM_PROMPTS_FACTOR" \
  -e SPLIT_TOTAL="${SPLIT_TOTAL:-0}" \
  -e NO_PROXY='*' -e no_proxy='*' -e HTTP_PROXY='' -e http_proxy='' -e HTTPS_PROXY='' -e https_proxy='' \
  --entrypoint bash "$IMAGE" -c '
    python3 -c "import pandas" 2>/dev/null || pip install --quiet pandas >/dev/null 2>&1
    bash "$REPO/glm5.1_notes/dual_replica_sync_driver.sh"
  ' 2>&1 | tee "$OUTDIR/driver.log"

# ---- 5. build the comparison table (vs single-replica 200584 baseline) --------
# Provide EITHER:
#   BASELINE_CSV  -- a single-replica 1P1D CSV (benchmark_parser.py schema), or
#   BASELINE_LOG  -- a single-replica benchmark_long_context *_CONCURRENCY.log
#                    (e.g. job 200584's); it is parsed to CSV here on the host.
# With a baseline you get the scaling_vs_single column; otherwise the table is
# just per-replica + aggregate.
BASELINE_CSV="${BASELINE_CSV:-}"
BASELINE_LOG="${BASELINE_LOG:-}"
if [ -z "$BASELINE_CSV" ] && [ -n "$BASELINE_LOG" ] && [ -f "$BASELINE_LOG" ]; then
  BASELINE_CSV="$OUTDIR/baseline_single.csv"
  echo "=== parsing single-replica baseline log -> $BASELINE_CSV ==="
  python3 "$REPO/benchmark_parser.py" "$BASELINE_LOG" --csv "$BASELINE_CSV" --no-screen \
    || { echo "WARN: baseline parse failed"; BASELINE_CSV=""; }
fi
echo "=== building comparison table ==="
python3 "$REPO/glm5.1_notes/compare_dual_replica.py" \
  --a "$OUTDIR/replicaA.csv" --b "$OUTDIR/replicaB.csv" \
  ${BASELINE_CSV:+--baseline "$BASELINE_CSV"} \
  -o "$OUTDIR/comparison.csv" 2>&1 | tee -a "$OUTDIR/driver.log"

echo "=== DONE. Results in $OUTDIR ==="
echo "  per-replica CSVs : replicaA.csv / replicaB.csv  (same schema as 200584)"
echo "  comparison       : comparison.csv"
# teardown() runs on EXIT
