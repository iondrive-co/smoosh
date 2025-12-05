#!/bin/bash
# Download YOLOv8n ONNX model for person detection

set -e

MODEL_DIR="src/main/resources/models"
MODEL_FILE="$MODEL_DIR/yolov8n.onnx"

echo "Downloading YOLOv8n ONNX model..."

mkdir -p "$MODEL_DIR"

# Try using Python ultralytics
if command -v python3 &> /dev/null; then
    echo "Attempting to use ultralytics package..."
    python3 << 'EOF'
try:
    from ultralytics import YOLO
    print("Downloading and exporting YOLOv8n model...")
    model = YOLO('yolov8n.pt')
    export_path = model.export(format='onnx')
    print(f"Model exported to: {export_path}")

    import shutil
    import os
    dest = "src/main/resources/models/yolov8n.onnx"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.move(export_path, dest)
    print(f"Model moved to: {dest}")
    print("✓ Model download successful!")
except ImportError:
    print("ultralytics not installed. Install with: pip install ultralytics")
    exit(1)
except Exception as e:
    print(f"Error: {e}")
    exit(1)
EOF
else
    echo "Error: Python3 not found"
    exit 1
fi
