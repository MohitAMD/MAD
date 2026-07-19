#!/usr/bin/env python3
"""
PR#47766 -- Complete sparse MLA metadata key fix for GLM-5.1-FP8.

Root cause: The sparse MLA metadata key was missing clamped_context_lens
and seg_lengths fields. With only 3 fields, the kernel reuses stale cached
metadata across decode steps when ISL > ~10K, causing the model to attend
to wrong KV positions and generate garbage output (repeating the prompt
template instead of answering).

Fix: Add clamped_context_lens.tobytes() and seg_lengths.tobytes() to make
the key unique per decode step at any context length.

Validated: v0.24.0 + this patch -> 10/10 DETERMINISTIC at 2K/8K/20K/96K,
Cohere NIAH score = 1.000.

(Vendored verbatim from build_glm51_v024_patched.sh, psakhamo@amd, for the
 WideEP-disagg image build. find_spec() locates vllm WITHOUT importing it, so
 this runs safely at Docker build time with no GPU.)
"""
import importlib.util, os, sys

vllm_root = os.path.dirname(os.path.dirname(importlib.util.find_spec('vllm').origin))
f = os.path.join(vllm_root, 'vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py')

with open(f) as fh:
    src = fh.read()

# Check if already applied
if 'clamped_context_lens.tobytes()' in src:
    print('[OK] PR#47766 already fully applied -- skipping')
    sys.exit(0)

old = '''        clamped_seq_lens = np.minimum(
            common_attn_metadata.seq_lens_cpu[:num_reqs].numpy(),
            self.topk_tokens,
        )
        metadata_key = (
            num_tokens,
            int(common_attn_metadata.max_query_len),
            self._num_attention_heads,
            clamped_seq_lens.tobytes(),
        )'''

new = '''        clamped_seq_lens = np.minimum(
            common_attn_metadata.seq_lens_cpu[:num_reqs].numpy(),
            self.topk_tokens,
        )
        seg_lengths = np.diff(
            np.asarray(common_attn_metadata.query_start_loc_cpu, dtype=np.int32)
        )
        clamped_context_lens = np.minimum(
            common_attn_metadata.seq_lens_cpu[:num_reqs].numpy() - seg_lengths,
            self.topk_tokens,
        )
        metadata_key = (
            num_tokens,
            int(common_attn_metadata.max_query_len),
            self._num_attention_heads,
            clamped_seq_lens.tobytes(),
            clamped_context_lens.tobytes(),
            seg_lengths.tobytes(),
        )'''

if old not in src:
    print('ERROR: Pattern not found in rocm_aiter_mla_sparse.py')
    print('       The base image may have a different version of this file.')
    sys.exit(1)

src = src.replace(old, new, 1)

with open(f, 'w') as fh:
    fh.write(src)

print('[OK] PR#47766 applied -- metadata key now has 6 fields')
print(f'     File: {f}')
