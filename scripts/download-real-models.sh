#!/usr/bin/env bash
set -euo pipefail

echo "📥 Downloading SMALL real model (bert-tiny) for Triton..."

MODEL_REPO="triton-server/model-repository"
MODEL_DIR="${MODEL_REPO}/text_classifier"
VERSION_DIR="${MODEL_DIR}/1"

rm -rf "${MODEL_DIR}"
mkdir -p "${VERSION_DIR}"

echo "🔧 Ensuring dependencies (transformers, torch, onnx)..."
python3 - <<'EOF'
import sys, subprocess
pkgs = ["torch", "transformers", "onnx"]
for p in pkgs:
    try:
        __import__(p)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", p])
EOF

echo "🧬 Exporting tiny model to ONNX..."
python3 - <<'EOF'
import os, torch, json
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn

model_name = "prajjwal1/bert-tiny"  # 2-layer tiny BERT
tokenizer = AutoTokenizer.from_pretrained(model_name)
base = AutoModel.from_pretrained(model_name)

class TinyClassifier(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        hidden = self.base.config.hidden_size
        self.classifier = nn.Linear(hidden, 2)
    def forward(self, input_ids, attention_mask):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state.mean(dim=1)
        return self.classifier(pooled)

model = TinyClassifier(base)
model.eval()

sample = tokenizer("hello tiny world", return_tensors="pt", truncation=True, padding="max_length", max_length=32)
os.makedirs("triton-server/model-repository/text_classifier/1", exist_ok=True)
torch.onnx.export(
    model,
    (sample["input_ids"], sample["attention_mask"]),
    "triton-server/model-repository/text_classifier/1/model.onnx",
    input_names=["input_ids","attention_mask"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0:"batch",1:"seq"},
                  "attention_mask": {0:"batch",1:"seq"},
                  "logits": {0:"batch"}},
    opset_version=14
)
with open("triton-server/model-repository/text_classifier/tokenizer_info.json","w") as f:
    json.dump({"max_length":32,"model":model_name}, f)
print("✅ Exported tiny ONNX model.")
EOF

echo "⚙️ Writing Triton config..."
cat > "${MODEL_DIR}/config.pbtxt" <<'EOF'
name: "text_classifier"
platform: "onnxruntime_onnx"
max_batch_size: 8
input [
  { name: "input_ids"      data_type: TYPE_INT64 dims: [ -1 ] },
  { name: "attention_mask" data_type: TYPE_INT64 dims: [ -1 ] }
]
output [
  { name: "logits" data_type: TYPE_FP32 dims: [ 2 ] }
]
default_model_filename: "model.onnx"
instance_group [{ kind: KIND_GPU }]
EOF

echo "📁 Model files:"
find "${MODEL_DIR}" -maxdepth 2 -type f -print

echo "✅ model ready."