import os
import sys
import torch
import joblib
import torch.nn as nn
import numpy as np
import cv2
import albumentations
import time
import cnn_models
import warnings

# Suppress PyTorch warnings related to future versions
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Validate input
if len(sys.argv) < 2:
    print("Usage: python test.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
if not os.path.exists(image_path):
    print(f"Error: Image file not found at {image_path}")
    sys.exit(1)

# Load label binarizer
lb_path = '../outputs/lb.pkl'
if not os.path.exists(lb_path):
    print(f"Error: Label binarizer file not found at {lb_path}")
    sys.exit(1)
lb = joblib.load(lb_path)

# Load model
model_path = '../outputs/model.pth'
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    sys.exit(1)

# Initialize the model
model = cnn_models.CustomCNN().to('cpu')

# Load the saved model weights
try:
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
except Exception as e:
    print(f"Error loading model: {str(e)}")
    sys.exit(1)

model.eval()

# Preprocessing
aug = albumentations.Compose([albumentations.Resize(224, 224)])

# Read and preprocess image
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not read image at {image_path}")
    sys.exit(1)

image_proc = aug(image=np.array(image))['image']
image_proc = np.transpose(image_proc, (2, 0, 1)).astype(np.float32)
image_tensor = torch.tensor(image_proc, dtype=torch.float).unsqueeze(0).to('cpu')

# Inference
start = time.time()
outputs = model(image_tensor)
_, preds = torch.max(outputs.data, 1)
predicted_label = lb.classes_[preds]

end = time.time()

# Output result
print(f"Prediction: {predicted_label}")
print(f"Inference Time: {(end - start):.3f}s")
