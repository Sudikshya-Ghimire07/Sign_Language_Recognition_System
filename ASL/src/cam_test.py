import torch
import joblib
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F
import time
import cnn_models
import threading
from torchvision import models
from gtts import gTTS
from playsound import playsound
import os
import uuid

# Load label binarizer and model
print("Loading label binarizer...")
lb = joblib.load('../outputs/lb.pkl')

model = cnn_models.CustomCNN()
model.load_state_dict(torch.load('../outputs/model.pth', map_location=torch.device('cpu')))
model.eval()
print('✅ Model and labels loaded successfully')

# ROI (Region of Interest) box coordinates
x1, y1, box_size = 100, 100, 200
x2, y2 = x1 + box_size, y1 + box_size

# Function to crop and preprocess hand area
def hand_area(img):
    hand = img[y1:y2, x1:x2]
    hand = cv2.resize(hand, (128, 128))
    return hand

# Non-blocking speech function using gTTS with unique filename
def speak_text(text):
    try:
        filename = "temp_audio.mp3"  # Use a fixed filename
        tts = gTTS(text=text, lang='en')
        tts.save(filename)
        playsound(filename)
        # No need to delete the file every time; it will be overwritten
    except Exception as e:
        print(f"❌ Error with TTS: {e}")

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('❌ Error: Unable to access camera.')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('../outputs/asl.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

# Status flags
tts_enabled = False
last_prediction = ""
last_speech_time = 0  # Timestamp to track when speech was last triggered
frame_count = 0
inference_interval = 5  # Perform inference every 5 frames

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Draw region box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 34, 255), 2)

    # Preprocess hand image
    hand = hand_area(frame)
    image = np.transpose(hand, (2, 0, 1)).astype(np.float32)  # Convert to CHW
    image = torch.tensor(image, dtype=torch.float).unsqueeze(0)  # Add batch dimension

    # Predict every N frames
    frame_count += 1
    if frame_count % inference_interval == 0:
        with torch.no_grad():
            outputs = model(image)
            _, preds = torch.max(outputs.data, 1)
            prediction_text = lb.classes_[preds.numpy()[0]]

        # Only trigger speech if the same prediction persists for more than 2 seconds
        if prediction_text != last_prediction:
            last_prediction = prediction_text
            last_speech_time = time.time()  # Reset the timer when prediction changes
        elif time.time() - last_speech_time > 2:  # Check if the prediction has been the same for 2 seconds
            if tts_enabled:
                print(f"🗣️ Speaking: {prediction_text}")
                tts_thread = threading.Thread(target=speak_text, args=(prediction_text,))
                tts_thread.start()
                last_speech_time = time.time()  # Reset the timer after speaking

    # Show prediction
    if 'prediction_text' in locals():
        cv2.putText(frame, f"Prediction: {prediction_text}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # TTS status display
    tts_status = "🗣️ TTS: ON" if tts_enabled else "🔇 TTS: OFF"
    color = (0, 255, 0) if tts_enabled else (0, 0, 255)
    cv2.putText(frame, tts_status, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show FPS
    fps = 1 / (time.time() - start_time + 1e-8)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Display frame
    cv2.imshow("ASL Real-Time Translator", frame)
    out.write(frame)

    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        tts_enabled = not tts_enabled
        print(f"TTS {'ENABLED' if tts_enabled else 'DISABLED'}")

cap.release()
cv2.destroyAllWindows()
