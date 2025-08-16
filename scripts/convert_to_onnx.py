import torch
from transformers import BertTokenizer, BertModel

# Load the PyTorch model
model_path = 'models/text_classifier/bert-base-uncased/pytorch_model.bin'
model = BertModel.from_pretrained('bert-base-uncased')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


# Convert the model to ONNX format
onnx_path = 'models/text_classifier/bert-base-uncased/model.onnx'
dummy_input = torch.randn(1, 128)
torch.onnx.export(model, dummy_input, onnx_path)

print('Model converted to ONNX format and saved to:', onnx_path)