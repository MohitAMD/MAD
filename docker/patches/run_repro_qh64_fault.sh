#!/bin/bash
# =============================================================================
# run_repro_qh64_fault.sh -- drive the aiter QH64 gfx942 MLA-decode fault repro
# on ONE MI300X (gfx942) node, showing (A) the fault with stock aiter (#3188)
# and (B) that patch_aiter_mla_qh64_fold.py makes it pass.
#
# Usage (from a login node that can `srun` / `docker run` on a gfx942 node):
#     bash run_repro_qh64_fault.sh [IMAGE]
# Default IMAGE carries aiter v0.1.18 (contains #3188):
#     vllm-disagg:glmv5.1-v0.25.1-pr47766-csfix-morishik-aiter0118
# =============================================================================
set -u
IMAGE="${1:-vllm-disagg:glmv5.1-v0.25.1-pr47766-csfix-morishik-aiter0118}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_in_container() {  # $1 = extra shell run before the repro
  docker run --rm --network host --ipc host \
    --device /dev/kfd --device /dev/dri --group-add video \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
    -v "$HERE":/repro --entrypoint bash "$IMAGE" -c "
      set -e
      $1
      python3 /repro/repro_aiter_mla_qh64_gfx942_fault.py
    "
  echo "exit=$?"
}

echo '############ A) STOCK aiter (#3188 present) -> EXPECT GPU FAULT ############'
run_in_container "true"

echo
echo '############ B) WITH patch_aiter_mla_qh64_fold.py -> EXPECT PASS ##########'
run_in_container "python3 /repro/patch_aiter_mla_qh64_fold.py"
