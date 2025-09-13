#!/usr/bin/env bash
set -euo pipefail

TRITON_URL="${TRITON_URL:-http://localhost:8000}"
TEXT="${1:-Hello from tiny model}"
MODEL_NAME="text_classifier"
READINESS_ENDPOINT="${TRITON_URL}/v2/models/${MODEL_NAME}/ready"

echo "Checking Triton model readiness for '${MODEL_NAME}'..."
for i in {1..30}; do
  if curl -fs "${READINESS_ENDPOINT}" >/dev/null 2>&1; then
    echo "Model is ready."
    break
  fi
  if [ "$i" -eq 30 ]; then
    echo "Model '${MODEL_NAME}' not ready after 30 attempts (60s)."
    exit 1
  fi
  sleep 2
done

echo "Ensuring tokenizer/runtime dependencies (host side)..."
python3 - <<'PY'
import sys, subprocess
for p in ("transformers","torch","requests","numpy"):
    try:
        __import__(p)
    except Exception:
        subprocess.check_call([sys.executable,"-m","pip","install","--quiet",p])
PY

echo "Sending inference request to Triton (${TRITON_URL})..."
python3 - "$TEXT" "$TRITON_URL" "$MODEL_NAME" <<'PY'
import sys, json, requests, numpy as np
from transformers import AutoTokenizer

text        = sys.argv[1]
triton_url  = sys.argv[2].rstrip("/")
model_name  = sys.argv[3]

tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
enc = tokenizer(
    text,
    return_tensors="np",
    truncation=True,
    padding="max_length",
    max_length=32
)

payload = {
  "inputs": [
    {
      "name": "input_ids",
      "datatype": "INT64",
      "shape": list(enc["input_ids"].shape),
      "data": enc["input_ids"].reshape(-1).tolist()
    },
    {
      "name": "attention_mask",
      "datatype": "INT64",
      "shape": list(enc["attention_mask"].shape),
      "data": enc["attention_mask"].reshape(-1).tolist()
    }
  ],
  "outputs": [ { "name": "logits" } ]
}

url = f"{triton_url}/v2/models/{model_name}/infer"
resp = requests.post(url, json=payload, timeout=60)
resp.raise_for_status()
data = resp.json()
print("Inference response:")
print(json.dumps(data, indent=2))

# Extract logits and softmax
outs = [o for o in data.get("outputs", []) if o.get("name") == "logits"]
if outs:
    flat = np.array(outs[0]["data"], dtype="float32").reshape(-1, 2)
    # Stable softmax
    ex = np.exp(flat - flat.max(axis=1, keepdims=True))
    probs = ex / ex.sum(axis=1, keepdims=True)
    labels = ["NEGATIVE","POSITIVE"]
    labeled = [{ "label": labels[int(np.argmax(row))], "probs": {labels[0]: float(row[0]), labels[1]: float(row[1])} } for row in probs]
    print("Probabilities:", probs.tolist())
    print("Labeled:", json.dumps(labeled, indent=2))
else:
    print("'logits' output not found.")
PY

echo "Real model inference complete."