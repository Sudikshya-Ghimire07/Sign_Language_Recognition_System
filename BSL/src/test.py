import os
import torch
import joblib
import torch.nn as nn
import numpy as np
import cv2
import albumentations
import time
import cnn_models

# Base input path containing folders A, B, C, etc.
IMAGE_BASE_PATH = '../input/test/'

# Load label binarizer
lb_path = '../outputs/lb.pkl'
if not os.path.exists(lb_path):
    print(f"Error: Label binarizer file not found at {lb_path}")
    exit()
lb = joblib.load(lb_path)
print("Label binarizer loaded.")

# Define image augmentation
aug = albumentations.Compose([albumentations.Resize(224, 224)])

# Load model
model_path = '../outputs/model.pth'
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    exit()

model = cnn_models.CustomCNN().to('cpu')
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()
print("Model loaded successfully.\n")

# Loop through each folder and process the single image inside
for folder in sorted(os.listdir(IMAGE_BASE_PATH)):
    folder_path = os.path.join(IMAGE_BASE_PATH, folder)
    if os.path.isdir(folder_path):
        # Find the first image in the folder
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not image_files:
            print(f"No image found in folder: {folder}")
            continue

        image_name = image_files[0]
        image_path = os.path.join(folder_path, image_name)
        print(f"Processing: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read the image at {image_path}")
            continue

        image_copy = image.copy()
        image = aug(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float).unsqueeze(0).to('cpu')

        # Inference
        start = time.time()
        outputs = model(image)
        _, preds = torch.max(outputs.data, 1)
        predicted_label = lb.classes_[preds]
        end = time.time()

        # Output results
        print(f"Predicted: {predicted_label} | Time: {(end - start):.3f}s")

        # Annotate and save image
        output_path = f"../outputs/{folder}_{image_name}"
        cv2.putText(image_copy, predicted_label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imwrite(output_path, image_copy)
        print(f"Saved: {output_path}\n")
