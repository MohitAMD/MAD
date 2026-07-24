#!/bin/bash
# Runs the updated cohere-accuracy-eval-suite INSIDE the eval container.
# Standalone (not an inline docker -c heredoc) so shell metacharacters in pip
# specs like 'pyarrow>=14,<18' and 'lm_eval[api]' aren't mangled by an outer shell.
# Expects env: EVAL_ENDPOINT, SERVED, ACCOUT (all passed via docker run -e).
set -x
cd /eval || exit 90
mkdir -p "$ACCOUT"
export PATH=/root/.local/bin:$PATH

echo "=== pip bootstrap ==="
python3 -m pip install --user -q --upgrade pip setuptools wheel 2>&1 | tail -5

echo "=== setup_evalscope (AA-LCR) ==="
bash scripts/setup_evalscope.sh 2>&1 | tail -15

echo "=== setup_livecodebench ==="
bash scripts/setup_livecodebench.sh 2>&1 | tail -15

echo "=== install lm_eval + pinned datasets (datasets<3 for RULER/aime scripts) ==="
python3 -m pip install --user -q --only-binary=:all: 'pyarrow>=14,<18' 2>&1 | tail -8
python3 -m pip install --user -q 'lm_eval[api]>=0.4.9' 'datasets<3.0.0' 'pebble>=5.1.0' 2>&1 | tail -30
# RULER niah_single_2 needs the lm_eval[ruler] extras (wonderwords + nltk) to build
# the haystack; without them niah hard-fails and skips the whole suite.
python3 -m pip install --user -q wonderwords nltk 2>&1 | tail -8
python3 -c "import nltk; [nltk.download(p, quiet=True) for p in ('punkt','punkt_tab','words')]" 2>&1 | tail -3

echo "=== import checks (lm_eval / datasets / evalscope versions) ==="
python3 -c 'import lm_eval, datasets, evalscope; print(lm_eval.__version__, datasets.__version__, evalscope.__version__)' 2>&1 | tail -5

echo "=== run_suite (glm51_suite.yaml, profile=pre_release, model=glm-5-1-fp8) ==="
python3 runners/run_suite.py \
  --suite-config glm51_suite.yaml \
  --profile pre_release \
  --model glm-5-1-fp8 \
  --endpoint "$EVAL_ENDPOINT" \
  --output-dir "$ACCOUT"
echo "run_suite rc=$?"
