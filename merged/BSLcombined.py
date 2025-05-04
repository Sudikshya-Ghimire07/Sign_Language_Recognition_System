import sys
import os
import cv2
import torch
import joblib
import time
import threading
import queue
import numpy as np
import pyttsx3
from keras.models import model_from_json, Sequential

# === PATH SETUP ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "..", "BSL", "src")
sys.path.append(SRC_DIR)

from cnn_models import CustomCNN

# === Load BSL Model ===
bsl_model = CustomCNN()
bsl_model.load_state_dict(torch.load(
    os.path.join(BASE_DIR, "..", "BSL", "outputs", "model.pth"),
    map_location=torch.device('cpu')
))
bsl_model.eval()
label_binarizer = joblib.load(os.path.join(BASE_DIR, "..", "BSL", "outputs", "lb.pkl"))

# === Load Facial Expression Model ===
with open(os.path.join(BASE_DIR, "..", "Facial_expression", "facialemotionmodel.json"), "r") as json_file:
    model_json = json_file.read()
face_model = model_from_json(model_json, custom_objects={'Sequential': Sequential})
face_model.load_weights(os.path.join(BASE_DIR, "..", "Facial_expression", "facialemotionmodel.h5"))

# === Constants ===
emotion_labels = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
    4: 'neutral', 5: 'sad', 6: 'surprise'
}
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# === TTS Setup with Queue ===
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_enabled = False
tts_queue = queue.Queue()

def tts_worker():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        tts_engine.say(text)
        tts_engine.runAndWait()
        tts_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

def speak(text):
    if tts_enabled and tts_queue.empty():
        tts_queue.put(text)

# === Helper Functions ===
def extract_hand(frame):
    hand = frame[100:324, 100:324]
    return cv2.resize(hand, (224, 224))

def preprocess_bsl(img):
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    return torch.tensor(img, dtype=torch.float).unsqueeze(0)

def preprocess_face(gray_img):
    img = gray_img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    return np.expand_dims(img, axis=0)

def log_event(label):
    print(f"[{time.strftime('%H:%M:%S')}] 🔍 {label}")

# === Main Loop ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

frame_count = 0
bsl_prediction = ""
emotion_prediction = ""
last_spoken_bsl = ""
last_spoken_emotion = ""
key_delay_counter = 0

print("✅ System ready. Press 'T' to toggle TTS | 'Q' or 'Esc' to quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # === BSL Prediction (every 5 frames) ===
    if frame_count % 5 == 0:
        try:
            hand_img = extract_hand(frame)
            hand_tensor = preprocess_bsl(hand_img)
            with torch.no_grad():
                bsl_output = bsl_model(hand_tensor)
            _, bsl_idx = torch.max(bsl_output.data, 1)
            bsl_prediction = label_binarizer.classes_[bsl_idx]

            if tts_enabled and bsl_prediction != last_spoken_bsl:
                speak(f"Sign {bsl_prediction}")
                last_spoken_bsl = bsl_prediction
                log_event(f"BSL: {bsl_prediction}")

        except Exception:
            bsl_prediction = "Error"

    # === Emotion Prediction (every 15 frames) ===
    if frame_count % 15 == 0:
        try:
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2]*f[3])  # largest face
                face_img = gray[y:y+h, x:x+w]
                if face_img.size > 0:
                    face_resized = cv2.resize(face_img, (48, 48))
                    face_input = preprocess_face(face_resized)
                    preds = face_model.predict(face_input, verbose=0)[0]
                    label_idx = int(np.argmax(preds))
                    emotion_prediction = emotion_labels[label_idx]

                    if tts_enabled and emotion_prediction != last_spoken_emotion:
                        speak(emotion_prediction)
                        last_spoken_emotion = emotion_prediction
                        log_event(f"Emotion: {emotion_prediction}")

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, emotion_prediction, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            else:
                emotion_prediction = "No Face"
        except Exception:
            emotion_prediction = "Error"

    # === Draw Hand ROI ===
    cv2.rectangle(frame, (100, 100), (324, 324), (0, 255, 255), 2)

    # === Display Info ===
    combined_text = f"BSL: {bsl_prediction} | Emotion: {emotion_prediction}"
    tts_status = "🗣️ ON" if tts_enabled else "🔇 OFF"
    cv2.putText(frame, combined_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"TTS: {tts_status}", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)

    cv2.imshow("BSL + Emotion Recognition", frame)

    # === Keyboard Control ===
    key = cv2.waitKey(1) & 0xFF
    if key in [ord('q'), 27]:
        print("👋 Exiting...")
        break
    elif key == ord('t') and key_delay_counter == 0:
        tts_enabled = not tts_enabled
        print(f"TTS {'ENABLED 🗣️' if tts_enabled else 'DISABLED 🔇'}")
        key_delay_counter = 10

    if key_delay_counter > 0:
        key_delay_counter -= 1

    frame_count += 1

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
tts_engine.stop()
tts_queue.put(None)
