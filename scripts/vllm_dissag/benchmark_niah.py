#!/usr/bin/env python3
# Needle-in-a-haystack long-context retrieval test.
# Adapted from vllm-project/vllm issue #47042 (GLM-5.2 sparse-MLA decode collapse),
# generalized to run against any OpenAI-compatible endpoint / model.
#
# Env:
#   NIAH_URL     endpoint (default http://127.0.0.1:30000/v1/chat/completions)
#   NIAH_MODEL   model name/tag the server serves (required — the served path)
#   NIAH_WORDS   comma list of context sizes in words (default 2000,8000,20000,35000)
#   NIAH_MAXTOK  max_tokens for the answer (default 2048)
#   NIAH_SEEDS   comma list of needle-layout seeds (default 0,1,2); summary reports
#                mean/min/max across seeds to separate real accuracy from variance
#   NIAH_TIMEOUT per-request timeout seconds (default 1800)
#   NIAH_WARMUP  1 (default) = send one throwaway request per context length BEFORE
#                scoring, so the first-hit JIT/kernel-autotune compile happens outside
#                the scored/gated window. On a freshly-booted node the first request of
#                a shape can take minutes to compile; without warmup that lands on the
#                first scored request -> false 0/10 or timeout. Warmup failures are
#                tolerated (logged, not fatal). Set 0 to disable.
import os, sys, json, random, urllib.request

URL = os.environ.get("NIAH_URL", "http://127.0.0.1:30000/v1/chat/completions")
MODEL = os.environ.get("NIAH_MODEL", "")
WORDS = [int(x) for x in os.environ.get("NIAH_WORDS", "2000,8000,20000,35000").split(",") if x.strip()]
MAXTOK = int(os.environ.get("NIAH_MAXTOK", "2048"))
TIMEOUT = float(os.environ.get("NIAH_TIMEOUT", "1800"))
# Needle layout is seeded, so a single run is deterministic (bit-exact repro on the
# same stack). Run multiple seeds to distinguish real accuracy from single-needle
# variance; the summary reports mean/min/max across seeds. Default 0,1,2.
SEEDS = [int(x) for x in os.environ.get("NIAH_SEEDS", "0,1,2").split(",") if x.strip()]
# NIAH_REPEAT>0: determinism probe -- run the SAME prompt (SEEDS[0]) N times instead
# of N different seeds, to directly measure run-to-run variance on an identical request
# (issue #47042's core claim). A non-deterministic stack yields different found-counts
# / different dropped needles across the repeats of one fixed prompt.
_REPEAT = int(os.environ.get("NIAH_REPEAT", "0"))
if _REPEAT > 0:
    SEEDS = [SEEDS[0]] * _REPEAT
WARMUP = os.environ.get("NIAH_WARMUP", "1") == "1"
# NIAH_THINKING: 0 (default) sends enable_thinking=False so the answer lands in
#   `content` with a small max_tokens (fast retrieval smoke test).
# 1 = FAITHFUL issue #47042 repro: leave thinking ON (do NOT set enable_thinking),
#   pair with a large NIAH_MAXTOK (e.g. 2048) so the long-context DECODE path is
#   actually exercised -- that decode is where the sparse-MLA collapse manifests.
#   Animals are scored from reasoning_content too, so CoT hits still count.
THINKING = os.environ.get("NIAH_THINKING", "0") == "1"
# NIAH_ENDPOINT: "chat" (default) -> /v1/chat/completions (applies the GLM chat
#   template + reasoning parser). "completions" -> /v1/completions with a raw prompt
#   (NO chat template, NO reasoning parser) to isolate the decode path from the
#   reasoning-parser machinery (matches the issue #47042 "no reasoning parser" note).
ENDPOINT = os.environ.get("NIAH_ENDPOINT", "chat").strip().lower()
# Warmup uses a generous timeout (cold compile of a long-context shape can take minutes)
# and never fails the run — its only job is to trigger compilation before scoring.
WARMUP_TIMEOUT = max(TIMEOUT, 1800.0)

FILLER = (
    "table chair window bottle pencil garden river mountain coffee planet "
    "engine guitar pillow ticket basket candle market silver button orange "
    "rocket napkin ladder pepper carpet helmet jacket mirror anchor pocket "
    "branch copper saddle tunnel violin wallet zipper meadow cactus pebble"
).split()
ANIMALS = ["elephant", "giraffe", "kangaroo", "penguin", "dolphin",
           "tiger", "rhinoceros", "octopus", "crocodile", "panda"]

SYSTEM = (
    "You read a word list and pick out the animals. Reply with a single "
    "comma-separated list of lowercase animal names. Output nothing else."
)


def make_haystack(n_words, seed=0):
    rng = random.Random(seed)
    words = [rng.choice(FILLER) for _ in range(n_words)]
    step = max(n_words // (len(ANIMALS) + 1), 1)
    for i, animal in enumerate(ANIMALS):
        words[min((i + 1) * step, len(words) - 1)] = animal
    return " ".join(words)


def _endpoint_url():
    """Resolve the request URL for the selected endpoint. For completions mode, rewrite
    a chat URL (.../v1/chat/completions) to the raw text URL (.../v1/completions)."""
    if ENDPOINT == "completions":
        return (URL.replace("/v1/chat/completions", "/v1/completions")
                   .replace("/chat/completions", "/completions"))
    return URL


def _request(n_words, seed, max_tokens, timeout):
    """POST one NIAH request; return (message_like_dict, error_str). Exactly one is
    non-None. The returned dict always exposes the generated text under 'content'
    (+ reasoning fields for the chat path) so run() can score it uniformly."""
    haystack = make_haystack(n_words, seed)
    if ENDPOINT == "completions":
        # Raw /v1/completions: no chat template, no reasoning parser. Fold the system
        # instruction + task into a single plain prompt.
        body = {
            "model": MODEL,
            "prompt": SYSTEM + "\n\nFind the animals in this list:\n\n" + haystack + "\n\nAnimals:",
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
    else:
        body = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Find the animals in this list:\n\n" + haystack},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        # Thinking models (e.g. GLM-5.1) emit chain-of-thought into a separate reasoning
        # field and leave `content` empty until the final answer; with a small max_tokens
        # the answer never appears in `content` and the score is a false 0/10. Default
        # (NIAH_THINKING=0) disables thinking so the answer lands in `content` directly.
        # NIAH_THINKING=1 leaves thinking ON to match issue #47042 (stresses the
        # long-context decode path where the sparse-MLA collapse shows up).
        if not THINKING:
            body["chat_template_kwargs"] = {"enable_thinking": False}
    data = json.dumps(body).encode()
    req = urllib.request.Request(_endpoint_url(), data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            choice = json.loads(r.read())["choices"][0]
            # Normalize: chat -> choice["message"]; completions -> {"content": choice["text"]}
            return (choice.get("message") if "message" in choice else {"content": choice.get("text", "")}), None
    except Exception as e:
        return None, str(e)


def run(n_words, seed, max_tokens):
    """Return (score, found_list, err). score/found are None on timeout/transport error
    (NOT a wrong answer); otherwise score is 0..10 and found is the retrieved animals."""
    msg, err = _request(n_words, seed, max_tokens, TIMEOUT)
    if err is not None:
        return None, None, err
    # Score content plus any reasoning field (some servers surface CoT as
    # `reasoning` or `reasoning_content`) so a thinking model is never mis-scored.
    text = ((msg.get("content") or "") + " "
            + (msg.get("reasoning_content") or "") + " "
            + (msg.get("reasoning") or "")).lower()
    found = sorted(a for a in ANIMALS if a in text)
    return len(found), found, None


# --- report header labels -----------------------------------------------------
# NIAH_TOPO: free-text hardware/topology label for the header (e.g. "MI300X 1P1D EP8").
TOPO = os.environ.get("NIAH_TOPO", "MI300X")
N = len(ANIMALS)

# NIAH_COMBOS: optional per-case "ISL/OSL,ISL/OSL" list (input words / output max_tokens).
# When set it OVERRIDES the flat NIAH_WORDS x NIAH_MAXTOK grid so each case carries its
# own decode budget -- needed for ISL/OSL shapes like 96000/32000 (96k-word context,
# 32k-token output cap). ISL ~= input tokens (1 filler word ~= 1 token).
_combos_raw = os.environ.get("NIAH_COMBOS", "").strip()
if _combos_raw:
    CASES = []
    for _pair in _combos_raw.split(","):
        _pair = _pair.strip()
        if not _pair:
            continue
        _isl, _sep, _osl = _pair.partition("/")
        CASES.append((int(_isl), int(_osl) if _osl.strip() else MAXTOK))
else:
    CASES = [(w, MAXTOK) for w in WORDS]


def _endpoint_line():
    # Reflects the actual request path + reasoning/thinking mode.
    try:
        from urllib.parse import urlparse
        path = urlparse(_endpoint_url()).path or _endpoint_url()
    except Exception:
        path = _endpoint_url()
    if ENDPOINT == "completions":
        mode = "raw prompt, no reasoning parser"
    else:
        mode = "thinking on" if THINKING else "thinking off (enable_thinking=False)"
    return "%s (%s)" % (path, mode)


def main():
    if not MODEL:
        print("NIAH_MODEL must be set (the served model path/name)", file=sys.stderr)
        sys.exit(2)
    model_short = os.path.basename(MODEL.rstrip("/")) or MODEL
    # Warmup FIRST (quietly) so the cold-compile of each shape happens off the scored
    # path and outside the clean report; only surface a warmup problem. One per unique ISL.
    if WARMUP:
        for n in sorted({isl for isl, _ in CASES}):
            _, err = _request(n, seed=0, max_tokens=8, timeout=WARMUP_TIMEOUT)
            if err is not None:
                print("  [warmup] words=%6d still compiling/err: %s" % (n, err), flush=True)

    bar = "=" * 65
    print(bar, flush=True)
    print("Repro: vllm-project/vllm#47042", flush=True)
    print("Model: %s | %s" % (model_short, TOPO), flush=True)
    print("Endpoint: %s" % _endpoint_line(), flush=True)
    print(bar, flush=True)

    for isl, osl in CASES:
        # ~1 token/word is a fine approximation for this filler-word haystack.
        print("\n--- ~%d words (~%d tok in / %d tok out max) ---" % (isl, isl, osl), flush=True)
        scores = []
        for i, s in enumerate(SEEDS, 1):  # trial index is 1-based
            score, found, err = run(isl, s, osl)
            if err is not None:
                print("  words=%6d  trial=%d  TIMEOUT/ERR  \u274c  %s" % (isl, i, err), flush=True)
                continue
            mark = "\u2705" if score == N else "\u274c"
            print("  words=%6d  trial=%d  found=%2d/%d  %s  %s"
                  % (isl, i, score, N, mark, found), flush=True)
            scores.append(score)
        if not scores:
            print("  Summary: NO-RESULT (all trials timed out/errored)", flush=True)
            continue
        mn, mx = min(scores), max(scores)
        if mn == mx:
            verdict = "\u2705 DETERMINISTIC" if mn == N else "\u26a0\ufe0f  DETERMINISTIC (but %d/%d)" % (mn, N)
        else:
            verdict = "\u26a0\ufe0f  NONDETERMINISTIC (\u0394=%d)" % (mx - mn)
        print("  Summary: min=%d max=%d %s" % (mn, mx, verdict), flush=True)

    print("\n" + bar, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
