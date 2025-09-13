#!/bin/bash

echo "📁 Creating real sample test data..."

# Create directory
mkdir -p test-data

# Create diverse text samples for testing
cat > test-data/text_samples.txt << 'EOF'
I absolutely love this product! It exceeded all my expectations and works perfectly.
This is the worst purchase I've ever made. Complete waste of money and time.
The weather is quite pleasant today with sunny skies and mild temperatures.
This movie is incredible! Amazing acting, great story, and beautiful cinematography.
Customer service was terrible. They were rude and unhelpful throughout the process.
The new restaurant downtown serves excellent food with great presentation.
I'm feeling neutral about this. It's neither good nor bad, just average.
This book changed my life! Highly recommend it to anyone interested in personal growth.
The software is buggy and crashes frequently. Very frustrating to use.
Beautiful sunset today. The colors in the sky are absolutely breathtaking.
EOF

# Create sample images using Python
python3 << 'EOF'
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# Create test-data directory
os.makedirs('test-data', exist_ok=True)

# Create sample images for testing
def create_test_image(filename, color, text=""):
    # Create a 224x224 image
    img = Image.new('RGB', (224, 224), color)
    
    if text:
        draw = ImageDraw.Draw(img)
        # Try to use a default font, fallback to basic if not available
        try:
            font = ImageFont.truetype("Arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Calculate text position (centered)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (224 - text_width) // 2
        y = (224 - text_height) // 2
        
        draw.text((x, y), text, fill='white', font=font)
    
    img.save(f'test-data/{filename}')
    print(f"✅ Created: test-data/{filename}")

# Create various test images
create_test_image('red_square.jpg', 'red', 'RED')
create_test_image('blue_circle.jpg', 'blue', 'BLUE')
create_test_image('green_triangle.jpg', 'green', 'GREEN')

# Create a more complex image (gradient)
def create_gradient_image():
    img_array = np.zeros((224, 224, 3), dtype=np.uint8)
    
    for i in range(224):
        for j in range(224):
            # Create a radial gradient
            center_x, center_y = 112, 112
            dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            normalized_dist = min(dist / 112, 1.0)
            
            # Rainbow gradient
            if normalized_dist < 0.33:
                img_array[i, j] = [255, int(255 * (1 - normalized_dist * 3)), 0]  # Red to yellow
            elif normalized_dist < 0.66:
                img_array[i, j] = [int(255 * (1 - (normalized_dist - 0.33) * 3)), 255, 0]  # Yellow to green
            else:
                img_array[i, j] = [0, int(255 * (1 - (normalized_dist - 0.66) * 3)), 255]  # Green to blue
    
    img = Image.fromarray(img_array)
    img.save('test-data/gradient.jpg')
    print("✅ Created: test-data/gradient.jpg")

create_gradient_image()

# Create a realistic test image (simple scene)
def create_scene_image():
    img = Image.new('RGB', (224, 224), 'lightblue')  # Sky
    draw = ImageDraw.Draw(img)
    
    # Ground
    draw.rectangle([0, 180, 224, 224], fill='green')
    
    # Sun
    draw.ellipse([170, 20, 200, 50], fill='yellow')
    
    # Clouds
    draw.ellipse([30, 40, 80, 70], fill='white')
    draw.ellipse([120, 30, 170, 60], fill='white')
    
    # Tree
    draw.rectangle([100, 120, 110, 180], fill='brown')  # Trunk
    draw.ellipse([85, 100, 125, 140], fill='darkgreen')  # Leaves
    
    img.save('test-data/simple_scene.jpg')
    print("✅ Created: test-data/simple_scene.jpg")

create_scene_image()

print("\n📊 Created test images:")
for img in ['red_square.jpg', 'blue_circle.jpg', 'green_triangle.jpg', 'gradient.jpg', 'simple_scene.jpg']:
    print(f"   • {img}")

EOF

# Create JSON test payloads
cat > test-data/text_inference_payload.json << 'EOF'
{
  "model_name": "text_classifier",
  "inputs": {
    "input_ids": {
      "data": [[101, 1045, 2293, 2023, 3155, 999, 102]],
      "shape": [1, 7],
      "datatype": "INT64"
    },
    "attention_mask": {
      "data": [[1, 1, 1, 1, 1, 1, 1]],
      "shape": [1, 7],
      "datatype": "INT64"
    }
  }
}
EOF

cat > test-data/image_inference_payload.json << 'EOF'
{
  "model_name": "resnet18",
  "inputs": {
    "input": {
      "data": [[[0.5, 0.6, 0.7]]],
      "shape": [1, 3, 224, 224],
      "datatype": "FP32"
    }
  }
}
EOF

# Create audio test file placeholder
echo "Creating audio test sample..." > test-data/sample_audio.wav

# Create batch test script
cat > test-data/batch_test.json << 'EOF'
{
  "batch_requests": [
    {
      "text": "I love this amazing product!",
      "expected_sentiment": "positive"
    },
    {
      "text": "This is terrible and I hate it.",
      "expected_sentiment": "negative"
    },
    {
      "text": "The weather is okay today.",
      "expected_sentiment": "neutral"
    }
  ]
}
EOF

echo "✅ Real test data created successfully!"
echo ""
echo "📁 Created files:"
echo "   • text_samples.txt - Diverse text samples for sentiment analysis"
echo "   • *.jpg - Test images for image classification"
echo "   • *.json - Pre-formatted inference payloads"
echo "   • batch_test.json - Batch testing configuration"
echo ""
echo "🧪 Usage:"
echo "   • Test with real data: ./scripts/test-real-inference.sh"
echo "   • Manual testing: curl -X POST http://localhost:8090/infer -d @test-data/text_inference_payload.json"
echo "   • Image testing: Use any .jpg file in test-data/"
