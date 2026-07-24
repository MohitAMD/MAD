#!/bin/bash
# Transparent rocminfo wrapper for GPU-less build nodes.
# Delegates to the real rocminfo (renamed *.real); only if that fails (no GPU
# present, e.g. during `docker build` on a login node) does it emit a canned
# gfx942/MI300X stanza so that aiter's import-time get_gfx_runtime() succeeds.
# On real GPU nodes at runtime the real rocminfo runs and true values are used.
REAL="$(command -v rocminfo).real"
# Fall back to the well-known ROCm path if the wrapper is invoked by realpath.
if [ ! -x "$REAL" ]; then
  REAL="$(dirname "$(readlink -f "$0")")/rocminfo.real"
fi
if [ -x "$REAL" ]; then
  o="$("$REAL" "$@" 2>/dev/null)"
  if [ $? -eq 0 ] && printf '%s' "$o" | grep -qi 'gfx'; then
    printf '%s\n' "$o"
    exit 0
  fi
fi
cat <<'ROCMINFO_EOF'
*******
Agent 2
*******
  Name:                    gfx942
  Marketing Name:          AMD Instinct MI300X
  Device Type:             GPU
  Compute Unit:            304
  Name:                    gfx942:sramecc+:xnack-
ROCMINFO_EOF
