#!/bin/bash
# Runs the cohere-accuracy-eval-suite (glm51_suite.yaml, profile=pre_release)
# against a DISAGG endpoint from inside the eval container.
#
# Disagg serves the model under its filesystem MODEL_PATH (no --served-model-name
# override in the launcher), so run_suite's default model id "glm-5-1-fp8" would
# 404 / be skipped (it filters on GET /models). We therefore remap the suite's
# model to the ACTUAL served name (SERVED) by cloning the glm-5-1-fp8 config
# entry (keeps the lcb.max_tokens / tokenizer_path / thinking fixes) under the
# served name, then run with --model "$SERVED".
#
# Expects env (via docker run -e): EVAL_ENDPOINT, SERVED, ACCOUT.
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

echo "=== install lm_eval + pinned datasets (datasets<3 for RULER/aime) + RULER extras ==="
python3 -m pip install --user -q --only-binary=:all: 'pyarrow>=14,<18' 2>&1 | tail -8
python3 -m pip install --user -q 'lm_eval[api]>=0.4.9' 'datasets<3.0.0' 'pebble>=5.1.0' 2>&1 | tail -30
python3 -m pip install --user -q wonderwords nltk 2>&1 | tail -8
python3 -c "import nltk; [nltk.download(p, quiet=True) for p in ('punkt','punkt_tab','words')]" 2>&1 | tail -3

echo "=== import checks ==="
python3 -c 'import lm_eval, datasets, evalscope; print(lm_eval.__version__, datasets.__version__, evalscope.__version__)' 2>&1 | tail -5

echo "=== remap suite model -> served disagg name: $SERVED ==="
# Work on a private copy so we never mutate the shared repo checkout.
rm -rf /tmp/es_run && cp -r /eval /tmp/es_run && cd /tmp/es_run
python3 - <<PY
import yaml
served = "$SERVED"
# 1) glm51_suite.yaml: point the 'glm' model role at the served name.
s = yaml.safe_load(open("config/glm51_suite.yaml"))
s["models"]["glm"]["name"] = served
yaml.safe_dump(s, open("config/glm51_suite.yaml", "w"), sort_keys=False)
# 2) models.yaml: clone the glm-5-1-fp8 entry under the served name (keeps
#    lcb.max_tokens=32768, lm_eval.tokenizer_path, chat_template_kwargs, etc.),
#    with model_id set to the served name so requests match the endpoint.
m = yaml.safe_load(open("config/models.yaml"))
base = m["models"].get("glm-5-1-fp8", {})
import copy
entry = copy.deepcopy(base)
entry["model_id"] = served
m["models"][served] = entry
yaml.safe_dump(m, open("config/models.yaml", "w"), sort_keys=False)
print("remapped glm role + models['%s'] (model_id=%s)" % (served, served))
PY

echo "=== run_suite (glm51_suite.yaml, profile=pre_release, model=$SERVED, disagg endpoint=$EVAL_ENDPOINT) ==="
python3 runners/run_suite.py \
  --suite-config glm51_suite.yaml \
  --profile pre_release \
  --model "$SERVED" \
  --endpoint "$EVAL_ENDPOINT" \
  --output-dir "$ACCOUT"
echo "run_suite rc=$?"
