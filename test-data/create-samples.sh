#!/bin/bash

echo "📁 Creating sample test data..."

# Create sample text file
echo "This is a sample text for classification testing. The AI system should process this text and return a classification result." > sample.txt

# Create sample image placeholder (would be a real image in production)
echo "Sample image data placeholder - replace with actual JPEG/PNG files" > sample.jpg

# Create sample audio placeholder (would be a real audio file in production)
echo "Sample audio data placeholder - replace with actual WAV/MP3 files" > sample.wav

# Create JSON test payload
cat > inference-request.json << EOF
{
  "model_name": "text-classifier",
  "inputs": [
    {
      "name": "INPUT_TEXT",
      "shape": [1],
      "datatype": "BYTES",
      "data": ["This is a test sentence for classification."]
    }
  ]
}
EOF

echo "✅ Test data created successfully!"
