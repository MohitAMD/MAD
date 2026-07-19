#!/usr/bin/env bash
# =============================================================================
# preflight_stage.sh - verify & stage prerequisites on every node of a SLURM
# allocation BEFORE launching the disagg run. Idempotent; safe to re-run.
#
# Why this exists:
#   - The private image is NOT pullable from the registry on this cluster, so a
#     fresh node has no image -> its container never starts -> the whole job
#     hangs forever at the "Waiting for nodes..." cluster-creation barrier.
#   - The 704 GiB model must live on node-local NVMe. Loading from NFS is ~25x
#     slower and overruns the proxy readiness timeout (boot dies before serving).
#   Fresh allocations land on new nodes lacking BOTH. Run this first to catch it.
#
# What it does, per node:
#   1. image   : ensure $IMAGE is present (docker save once from a node/host that
#                has it -> NFS tar -> docker load on the nodes missing it).
#   2. model   : ensure $MODEL_LOCAL_DIR/$MODEL_NAME exists locally (cp -a from
#                $MODEL_NFS_SRC inside a root container; the dir is root-owned).
#   3. health  : (optional, --health) flag nodes whose GPUs fail amdsmi ASIC
#                query == RAS-disabled hardware (the silent boot-killer).
#
# Usage (from inside your salloc, or pass --jobid):
#   bash preflight_stage.sh                 # check + stage image & model
#   bash preflight_stage.sh --health        # also run GPU/RAS health probe
#   bash preflight_stage.sh --check-only    # report only, stage nothing
#   bash preflight_stage.sh --jobid 160455  # target a specific job
#
# Env overrides (defaults match the GLM-5.1 workflow):
#   IMAGE  MODEL_NAME  MODEL_NFS_SRC  MODEL_LOCAL_DIR  TAR_DIR
#
# Exit code: 0 = all nodes ready; 1 = something still missing/unhealthy.
# =============================================================================
set -uo pipefail

IMAGE="${IMAGE:-rocmshared/pytorch-private:glm5.1-47766-shiklatest_prewarmed_}"
MODEL_NAME="${MODEL_NAME:-GLM-5.1-FP8}"
MODEL_NFS_SRC="${MODEL_NFS_SRC:-/shared_inference/models_blog/${MODEL_NAME}}"
MODEL_LOCAL_DIR="${MODEL_LOCAL_DIR:-/mnt/m2m_nobackup/models_blog}"
TAR_DIR="${TAR_DIR:-/shared_inference/mdeopuja/model_blog_logs}"

JOBID="${SLURM_JOB_ID:-}"
DO_HEALTH=0
DO_STAGE=1

while [ $# -gt 0 ]; do
  case "$1" in
    --jobid)      JOBID="$2"; shift 2;;
    --health)     DO_HEALTH=1; shift;;
    --check-only) DO_STAGE=0; shift;;
    -h|--help)    sed -n '2,40p' "$0"; exit 0;;
    *) echo "unknown arg: $1 (try --help)" >&2; exit 2;;
  esac
done

log()  { printf '\033[1m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\033[33m[%s] WARN:\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
err()  { printf '\033[31m[%s] ERROR:\033[0m %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

[ -n "$JOBID" ] || { err "no SLURM job id; run inside an salloc or pass --jobid <id>"; exit 2; }

# --- resolve allocation nodes -------------------------------------------------
NODELIST_RAW="$(squeue -j "$JOBID" -h -o '%N' 2>/dev/null)"
[ -n "$NODELIST_RAW" ] || { err "job $JOBID not found / not running"; exit 2; }
mapfile -t NODES < <(scontrol show hostnames "$NODELIST_RAW" 2>/dev/null)
[ "${#NODES[@]}" -gt 0 ] || { err "could not resolve nodes for job $JOBID"; exit 2; }
NODES_CSV="$(IFS=,; echo "${NODES[*]}")"
N=${#NODES[@]}
log "Job $JOBID: $N nodes -> $NODES_CSV"

# --- working dir for remote helper scripts (on shared FS) ---------------------
WORK="${TAR_DIR}/.preflight_${JOBID}"
mkdir -p "$WORK" || { err "cannot create work dir $WORK"; exit 2; }
TAR="${WORK}/image.tar"
export IMAGE MODEL_NAME MODEL_NFS_SRC MODEL_LOCAL_DIR TAR

SRUN="srun --jobid=${JOBID} --overlap --ntasks-per-node=1"
# run_on <csv> <count> <script-path>   -> runs a helper on the given node subset
run_on() { $SRUN --nodes="$2" --ntasks="$2" --nodelist="$1" bash "$3" 2>&1; }

# --- clear STALE aiter JIT build locks (pre-launch) ---------------------------
# aiter's FileBaton.wait() spins forever on a lock file with NO timeout, and
# baton.release() only runs in a `finally` -> a SIGKILL'd job (every scancel) leaves
# lock_module_* files behind. The next run's 8 local workers all see the lock, all
# wait(), and DEADLOCK ("waiting for baton release", observed job 200011 module_mla_reduce,
# 202799 module_moe_fmoe_asm, 202959 module_rmsnorm). At preflight time no server is
# running yet, so every such lock is stale and safe to delete; built .so's are kept.
#
# CRITICAL: the recipe maps the PERSISTENT node-local cache
# /mnt/m2m_nobackup/*/vllm_jit_cache/<digest> -> /opt/vllm_cache (AITER_JIT_DIR=
# /opt/vllm_cache/aiter_jit), so the real locks live under .../vllm_jit_cache/*/
# aiter_jit/build/. The old globs (/tmp/vllm_cache_*, /root/.aiter) MISSED that path,
# which is why the deadlocks kept recurring. Cover all known cache roots.
# The lock files are created by the container (running as ROOT), so they are
# root:root owned. Clearing them from the host as the unprivileged SLURM user fails
# with "Permission denied" (the bug that let stale locks persist and deadlock every
# run: FileBaton.wait() spins forever on a lock no live process will release). So do
# the rm INSIDE a root container that bind-mounts the cache roots. $IMAGE / $h expand
# at node runtime (IMAGE is exported; same pattern as the model-staging helpers).
cat > "$WORK/clear_aiter_locks.sh" <<'EOS'
h="$(hostname)"
docker run --rm \
  -v /mnt/m2m_nobackup:/mnt/m2m_nobackup \
  -v /tmp:/tmp \
  --entrypoint bash "$IMAGE" -c '
    n=0
    for L in /mnt/m2m_nobackup/*/vllm_jit_cache/*/aiter_jit/build/lock_* \
             /tmp/vllm_cache_*/aiter_jit/build/lock_* \
             /root/.aiter/build/*/lock; do
      [ -e "$L" ] || continue
      rm -rf "$L" 2>/dev/null && n=$((n+1))
    done
    echo "cleared_stale_locks=$n"
  ' 2>/dev/null | sed "s/^/$h /"
EOS
log "Clearing stale aiter JIT build locks on all nodes..."
run_on "$NODES_CSV" "$N" "$WORK/clear_aiter_locks.sh" | sed 's/^/    /'

# =============================================================================
# 1) IMAGE
# =============================================================================
cat > "$WORK/chk_image.sh" <<'EOS'
docker image inspect "$IMAGE" >/dev/null 2>&1 && echo "$(hostname) ok" || echo "$(hostname) missing"
EOS
cat > "$WORK/load_image.sh" <<'EOS'
docker load -i "$TAR" >/dev/null 2>&1 && echo "$(hostname) loaded" || echo "$(hostname) LOAD_FAILED"
EOS

log "Checking image '$IMAGE' on all nodes..."
IMG_OK=(); IMG_MISS=()
while read -r node st; do
  [ -z "$node" ] && continue
  if [ "$st" = ok ]; then IMG_OK+=("$node"); else IMG_MISS+=("$node"); fi
done < <(run_on "$NODES_CSV" "$N" "$WORK/chk_image.sh" | sort)
log "  image present: ${#IMG_OK[@]}/$N   missing: ${IMG_MISS[*]:-none}"

IMG_RESULT="ok"
if [ "${#IMG_MISS[@]}" -gt 0 ]; then
  if [ "$DO_STAGE" -eq 0 ]; then
    IMG_RESULT="missing(${#IMG_MISS[@]})"
  else
    # pick a source that already has the image: an alloc node, else this host
    if [ "${#IMG_OK[@]}" -gt 0 ]; then
      SRC="${IMG_OK[0]}"
      log "Saving image from $SRC -> $TAR (~minutes)"
      printf 'docker save "$IMAGE" -o "$TAR" && echo saved || echo SAVE_FAILED\n' > "$WORK/save_image.sh"
      run_on "$SRC" 1 "$WORK/save_image.sh"
    elif docker image inspect "$IMAGE" >/dev/null 2>&1; then
      log "No alloc node has the image; saving from local host -> $TAR"
      docker save "$IMAGE" -o "$TAR"
    else
      err "image '$IMAGE' not on any alloc node nor local host; cannot stage it."
      err "  -> load it somewhere first (docker load), then re-run."
      IMG_RESULT="UNSTAGEABLE"
    fi
    if [ -f "$TAR" ]; then
      CSV="$(IFS=,; echo "${IMG_MISS[*]}")"
      log "Loading image on ${#IMG_MISS[@]} node(s): $CSV"
      run_on "$CSV" "${#IMG_MISS[@]}" "$WORK/load_image.sh"
      rm -f "$TAR"
      IMG_RESULT="staged"
    fi
  fi
fi

# =============================================================================
# 2) MODEL (local NVMe)
# =============================================================================
SRC_FILES="$(ls "$MODEL_NFS_SRC" 2>/dev/null | wc -l)"
[ "$SRC_FILES" -gt 0 ] || { err "model source '$MODEL_NFS_SRC' is empty/unreadable"; exit 2; }
export SRC_FILES

cat > "$WORK/chk_model.sh" <<'EOS'
d="$MODEL_LOCAL_DIR/$MODEL_NAME"
n="$(ls "$d" 2>/dev/null | wc -l)"
echo "$(hostname) $n"
EOS
# stage runs as root in a container (the model_blog dir is root/ravgupta-owned).
# rm the dst first so a partial/incomplete copy is replaced cleanly.
cat > "$WORK/stage_model.sh" <<'EOS'
h="$(hostname)"
docker run --rm \
  -v /shared_inference:/shared_inference \
  -v /mnt/m2m_nobackup:/mnt/m2m_nobackup \
  --entrypoint bash "$IMAGE" -c "
    mkdir -p '$MODEL_LOCAL_DIR' &&
    rm -rf '$MODEL_LOCAL_DIR/$MODEL_NAME' &&
    cp -a '$MODEL_NFS_SRC' '$MODEL_LOCAL_DIR/'
  " && echo "$h model_done" || echo "$h MODEL_FAILED"
EOS

log "Checking local model '$MODEL_LOCAL_DIR/$MODEL_NAME' (expect $SRC_FILES files)..."
MDL_OK=(); MDL_MISS=()
while read -r node cnt; do
  [ -z "$node" ] && continue
  if [ "$cnt" = "$SRC_FILES" ]; then MDL_OK+=("$node"); else MDL_MISS+=("$node"); fi
done < <(run_on "$NODES_CSV" "$N" "$WORK/chk_model.sh" | sort)
log "  model present: ${#MDL_OK[@]}/$N   missing/incomplete: ${MDL_MISS[*]:-none}"

MDL_RESULT="ok"
if [ "${#MDL_MISS[@]}" -gt 0 ]; then
  if [ "$DO_STAGE" -eq 0 ]; then
    MDL_RESULT="missing(${#MDL_MISS[@]})"
  else
    CSV="$(IFS=,; echo "${MDL_MISS[*]}")"
    log "Staging model on ${#MDL_MISS[@]} node(s): $CSV"
    log "  (~705 GiB/node; tens of minutes. Runs in parallel.)"
    run_on "$CSV" "${#MDL_MISS[@]}" "$WORK/stage_model.sh"
    MDL_RESULT="staged"
  fi
fi

# =============================================================================
# 3) HEALTH (optional): amdsmi ASIC query == GPU not RAS-disabled
# =============================================================================
HEALTH_BAD=()
if [ "$DO_HEALTH" -eq 1 ]; then
  cat > "$WORK/amdsmi_probe.py" <<'PY'
import socket, amdsmi
h = socket.gethostname()
amdsmi.amdsmi_init()
hs = amdsmi.amdsmi_get_processor_handles()
ok = bad = 0
for x in hs:
    try:
        amdsmi.amdsmi_get_gpu_asic_info(x); ok += 1
    except Exception:
        bad += 1
print(f"{h} gpus={len(hs)} ok={ok} bad={bad}")
amdsmi.amdsmi_shut_down()
PY
  cat > "$WORK/probe_health.sh" <<'EOS'
docker run --rm --device /dev/dri --device /dev/kfd --group-add video --privileged \
  -v /shared_inference:/shared_inference --entrypoint python3 "$IMAGE" \
  "$TAR_DIR/.preflight_$SLURM_JOB_ID/amdsmi_probe.py" 2>/dev/null || echo "$(hostname) PROBE_FAILED"
EOS
  export TAR_DIR
  log "Running GPU/RAS health probe on all nodes..."
  while read -r node rest; do
    [ -z "$node" ] && continue
    echo "    $node $rest"
    case "$rest" in
      *bad=0) : ;;
      *) HEALTH_BAD+=("$node");;
    esac
  done < <(run_on "$NODES_CSV" "$N" "$WORK/probe_health.sh" | sort)
fi

# =============================================================================
# Re-verify + summary
# =============================================================================
log "Re-verifying after staging..."
FINAL_IMG_MISS=(); FINAL_MDL_MISS=()
while read -r node st;  do [ -n "$node" ] && [ "$st"  != ok ]          && FINAL_IMG_MISS+=("$node"); done < <(run_on "$NODES_CSV" "$N" "$WORK/chk_image.sh" | sort)
while read -r node cnt; do [ -n "$node" ] && [ "$cnt" != "$SRC_FILES" ] && FINAL_MDL_MISS+=("$node"); done < <(run_on "$NODES_CSV" "$N" "$WORK/chk_model.sh" | sort)

rm -rf "$WORK"

echo
echo "================ PRE-FLIGHT SUMMARY (job $JOBID, $N nodes) ================"
printf '  image  : %s\n' "$([ "${#FINAL_IMG_MISS[@]}" -eq 0 ] && echo "READY on all $N nodes" || echo "MISSING on: ${FINAL_IMG_MISS[*]}")"
printf '  model  : %s\n' "$([ "${#FINAL_MDL_MISS[@]}" -eq 0 ] && echo "READY on all $N nodes" || echo "MISSING/incomplete on: ${FINAL_MDL_MISS[*]}")"
if [ "$DO_HEALTH" -eq 1 ]; then
  printf '  health : %s\n' "$([ "${#HEALTH_BAD[@]}" -eq 0 ] && echo "all GPUs OK" || echo "RAS-FAULTED (reboot/avoid): ${HEALTH_BAD[*]}")"
fi
echo "=========================================================================="

if [ "${#FINAL_IMG_MISS[@]}" -eq 0 ] && [ "${#FINAL_MDL_MISS[@]}" -eq 0 ] && [ "${#HEALTH_BAD[@]}" -eq 0 ]; then
  log "All prerequisites satisfied. Safe to launch."
  exit 0
else
  warn "Not all nodes are ready. Fix the above (exclude faulted nodes / re-run) before launching."
  exit 1
fi
