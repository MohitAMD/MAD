"""
Startup fix for GLM-5.2-FP8 on recipe15 image.

Bugs fixed:
  1. vllm.envs missing Q/K/V_SCALE_CONSTANT
  2. moriio_common missing fold_local_rank / pod_index
  3. scheduler.py KeyError on MoRIIO compound req_ids

All prints go to stderr to avoid contaminating compiler stdout.
"""
import os
import sys


def _fix_envs():
    try:
        import vllm.envs as _envs_mod
        ev = _envs_mod.environment_variables
        if "Q_SCALE_CONSTANT" not in ev:
            ev["Q_SCALE_CONSTANT"] = lambda: int(os.getenv("Q_SCALE_CONSTANT", "200"))
            ev["K_SCALE_CONSTANT"] = lambda: int(os.getenv("K_SCALE_CONSTANT", "200"))
            ev["V_SCALE_CONSTANT"] = lambda: int(os.getenv("V_SCALE_CONSTANT", "100"))
            print("[glm52-fix] patched Q/K/V_SCALE_CONSTANT", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[glm52-fix] WARNING: envs patch failed: {e}", file=sys.stderr, flush=True)


def _fix_moriio_common():
    try:
        import vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common as _mc
        if not hasattr(_mc, "fold_local_rank"):
            def fold_local_rank(global_rank, dp_local):
                return global_rank % dp_local if dp_local > 0 else global_rank
            def pod_index(global_rank, dp_local):
                return global_rank // dp_local if dp_local > 0 else 0
            _mc.fold_local_rank = fold_local_rank
            _mc.pod_index = pod_index
            print("[glm52-fix] patched fold_local_rank + pod_index", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[glm52-fix] WARNING: moriio_common patch failed: {e}", file=sys.stderr, flush=True)


def _fix_scheduler_keyerror():
    """Guard scheduler.py against KeyError when MoRIIO compound req_id not in req_id_to_index."""
    try:
        import vllm.v1.core.sched.scheduler as _sched_mod
        path = _sched_mod.__file__
        if path.endswith(".pyc"):
            path = path[:-1]

        OLD = "            req_index = model_runner_output.req_id_to_index[req_id]"
        NEW = (
            "            if req_id not in model_runner_output.req_id_to_index:\n"
            "                continue  # glm52-fix: MoRIIO synthetic req not in index\n"
            "            req_index = model_runner_output.req_id_to_index[req_id]"
        )

        with open(path) as f:
            code = f.read()

        if "glm52-fix: MoRIIO synthetic" in code:
            return  # already patched

        if OLD not in code:
            print("[glm52-fix] WARNING: scheduler anchor not found -- skipping", file=sys.stderr, flush=True)
            return

        with open(path, "w") as f:
            f.write(code.replace(OLD, NEW, 1))

        import importlib
        importlib.reload(vllm.v1.core.sched.scheduler)
        print("[glm52-fix] patched scheduler.py KeyError guard", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[glm52-fix] WARNING: scheduler patch failed: {e}", file=sys.stderr, flush=True)


_fix_envs()
_fix_moriio_common()
_fix_scheduler_keyerror()


def _fix_reqmeta_multi_pod_hosts():
    """Add missing multi_pod_hosts and remote_dp_size_local fields to ReqMeta dataclass.

    moriio_connector.py accesses meta.multi_pod_hosts and meta.remote_dp_size_local
    but ReqMeta in moriio_common.py doesn't define these fields → AttributeError
    at first KV transfer (disagg path only).
    """
    try:
        import vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common as _mc
        from dataclasses import fields as dc_fields
        field_names = {f.name for f in dc_fields(_mc.ReqMeta)}
        if "multi_pod_hosts" in field_names:
            return  # already patched

        # Patch: add missing optional fields with safe defaults
        _mc.ReqMeta.multi_pod_hosts = None       # list[str] | None
        _mc.ReqMeta.remote_dp_size_local = 0  # int | None; 0 means "unknown, use remote_dp_size"
        print("[glm52-fix] patched ReqMeta with multi_pod_hosts + remote_dp_size_local", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[glm52-fix] WARNING: ReqMeta patch failed: {e}", file=sys.stderr, flush=True)


_fix_reqmeta_multi_pod_hosts()


def _fix_moriio_mtp_block_offsets():
    """MTP + WideEP PD-disagg KV-transfer fix.

    With speculative decoding (MTP, num_speculative_tokens=N) the PREFILL
    (producer) KV-cache manager reserves N lookahead slots, so its
    local_block_ids is longer than the DECODE (consumer) remote_block_ids
    (which covers the prompt only) by exactly N. With block_size=1 the block
    order is positional, so local[:len(remote)] is the real prompt KV and
    local[len(remote):] are empty lookahead scratch that decode does not
    allocate. Stock moriio_layout.compute_block_transfer_offsets raises
    "local_block_ids longer than remote_block_ids: 8 > 5" on this. Wrap it to
    transfer only the prompt prefix (drop the trailing lookahead blocks),
    which is exactly what the function's own docstring already does for the
    symmetric shorter-local case.

    Patched at moriio_layout level: the connector does
    `from ...moriio_layout import compute_block_transfer_offsets`, so as long
    as this runs before the connector is imported (sitecustomize => yes) the
    connector binds the wrapped fn. If the connector is already loaded we
    rebind its module attribute too.
    """
    try:
        import functools
        import vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_layout as _ly
        _orig = _ly.compute_block_transfer_offsets
        if getattr(_orig, "_glm_mtp_wrapped", False):
            return

        _warned = {"n": 0}

        @functools.wraps(_orig)
        def _wrapped(*args, **kwargs):
            try:
                if "local_block_ids" in kwargs:
                    lb = kwargs["local_block_ids"]
                else:
                    lb = args[3] if len(args) > 3 else None
                if "remote_block_ids" in kwargs:
                    rb = kwargs["remote_block_ids"]
                else:
                    rb = args[4] if len(args) > 4 else None
                if lb is not None and rb is not None and len(lb) > len(rb):
                    if _warned["n"] < 8:
                        print(
                            f"[glm52-fix][mtp] clamp local_block_ids {len(lb)}->{len(rb)} "
                            f"(drop {len(lb) - len(rb)} trailing spec-lookahead blocks)",
                            file=sys.stderr, flush=True,
                        )
                        _warned["n"] += 1
                    lb2 = list(lb)[: len(rb)]
                    if "local_block_ids" in kwargs:
                        kwargs["local_block_ids"] = lb2
                    else:
                        args = args[:3] + (lb2,) + args[4:]
            except Exception as _e:
                print(f"[glm52-fix][mtp] WARNING: clamp guard error: {_e}", file=sys.stderr, flush=True)
            return _orig(*args, **kwargs)

        _wrapped._glm_mtp_wrapped = True
        _ly.compute_block_transfer_offsets = _wrapped

        _cc_name = "vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector"
        _cc = sys.modules.get(_cc_name)
        if _cc is not None and hasattr(_cc, "compute_block_transfer_offsets"):
            _cc.compute_block_transfer_offsets = _wrapped
        print("[glm52-fix] patched compute_block_transfer_offsets for MTP spec-lookahead blocks", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[glm52-fix] WARNING: moriio MTP block-offset patch failed: {e}", file=sys.stderr, flush=True)


_fix_moriio_mtp_block_offsets()
