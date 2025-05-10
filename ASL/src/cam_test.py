import torch
import joblib
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F
import time
import cnn_models
import sys
import pyttsx3
import threading  # ✅ For async TTS
from torchvision import models

# Load label binarizer and model
lb = joblib.load('../outputs/lb.pkl')
model = cnn_models.CustomCNN()
model.load_state_dict(torch.load('../outputs/model.pth', map_location=torch.device('cpu')))
model.eval()

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

print('✅ Model loaded successfully')

# Function to extract hand area from frame
def hand_area(img):
    hand = img[100:324, 100:324]
    hand = cv2.resize(hand, (128, 128))  # 🔧 Resize for better performance
    return hand

# TTS function (non-blocking)
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# Start webcam capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('❌ Error: Unable to access camera.')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('session_temp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

# Flags and tracking
tts_enabled = False
last_prediction = ""
frame_count = 0
inference_interval = 5  # 🔧 Predict every 5 frames

# Track duration of current prediction
stable_start_time = None
spoken_flag = False

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Draw rectangle on hand area
    cv2.rectangle(frame, (100, 100), (324, 324), (20, 34, 255), 2)
    hand = hand_area(frame)

    # Preprocess hand image
    image = np.transpose(hand, (2, 0, 1)).astype(np.float32)
    image = torch.tensor(image, dtype=torch.float).unsqueeze(0)

    # Predict every N frames only
    frame_count += 1
    if frame_count % inference_interval == 0:
        with torch.no_grad():
            outputs = model(image)
            _, preds = torch.max(outputs.data, 1)
            prediction_text = lb.classes_[preds]

        current_time = time.time()

        if prediction_text == last_prediction:
            if stable_start_time is None:
                stable_start_time = current_time
                spoken_flag = False
            elif current_time - stable_start_time >= 0.5 and not spoken_flag and tts_enabled:
                print(f"🗣️ Speaking: {prediction_text}")
                tts_thread = threading.Thread(target=speak_text, args=(prediction_text,))
                tts_thread.start()
                spoken_flag = True
        else:
            last_prediction = prediction_text
            stable_start_time = time.time()
            spoken_flag = False

        if not tts_enabled:
            spoken_flag = False
            stable_start_time = None

    # Display prediction
    if 'prediction_text' in locals():
        cv2.putText(frame, f"Prediction: {prediction_text}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display TTS status
    tts_status = "🗣️ TTS: ON" if tts_enabled else "🔇 TTS: OFF"
    color = (0, 255, 0) if tts_enabled else (0, 0, 255)
    cv2.putText(frame, tts_status, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # FPS
    fps = 1 / (time.time() - start_time + 1e-6)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Show frame
    cv2.imshow("ASL Real-Time Translator", frame)
    out.write(frame)



    # Keypress control
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        tts_enabled = not tts_enabled
        print(f"TTS {'ENABLED' if tts_enabled else 'DISABLED'}")

cap.release()
cv2.destroyAllWindows()




# Safely stop the TTS engine before program exit
engine.stop()


