import sys
import os
import cv2
import torch
import joblib
import numpy as np
import pyttsx3
import threading
from keras.models import model_from_json, Sequential

# === PATH SETUP ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "..", "ASL", "src")
sys.path.append(SRC_DIR)

# === IMPORT CUSTOM ASL MODEL ===
from cnn_models import CustomCNN

# === LOAD MODELS ===
# ASL
asl_model = CustomCNN()
asl_model.load_state_dict(torch.load(
    os.path.join(BASE_DIR, "..", "ASL", "outputs", "model.pth"),
    map_location=torch.device('cpu')
))
asl_model.eval()

label_binarizer = joblib.load(
    os.path.join(BASE_DIR, "..", "ASL", "outputs", "lb.pkl")
)

# Facial Expression Model
with open(os.path.join(BASE_DIR, "..", "Facial_expression", "facialemotionmodel.json"), "r") as json_file:
    model_json = json_file.read()

face_model = model_from_json(model_json, custom_objects={'Sequential': Sequential})
face_model.load_weights(os.path.join(BASE_DIR, "..", "Facial_expression", "facialemotionmodel.h5"))

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotion_labels = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
    4: 'neutral', 5: 'sad', 6: 'surprise'
}

# === TTS Setup ===
tts_enabled = False
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

def speak_in_thread(text):
    threading.Thread(target=lambda: tts_engine.say(text) or tts_engine.runAndWait()).start()

# === Helper Functions ===
def extract_hand(frame):
    hand = frame[100:324, 100:324]
    return cv2.resize(hand, (224, 224))

def extract_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces, gray

def preprocess_asl_image(img):
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    return torch.tensor(img, dtype=torch.float).unsqueeze(0)

def preprocess_face_image(img):
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

# === MAIN LOOP ===
cap = cv2.VideoCapture(0)

last_spoken_asl = ""
last_spoken_emotion = ""
key_delay_counter = 0

print("📸 Starting ASL + Facial Expression Detection...")
print("🔁 Press 'T' to toggle TTS | Press 'Q' to Quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    asl_text = "ASL: Detecting..."
    emotion_text = "Emotion: Detecting..."

    # Draw hand ROI
    cv2.rectangle(frame, (100, 100), (324, 324), (0, 255, 255), 2)

    # === FACIAL EXPRESSION ===
    try:
        faces, gray = extract_face(frame)
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                if face_img.size == 0:
                    continue

                face_img = cv2.resize(face_img, (48, 48))
                face_input = preprocess_face_image(face_img)
                preds = face_model.predict(face_input, verbose=0)
                preds = np.exp(preds[0]) / np.sum(np.exp(preds[0]))
                emotion_label = emotion_labels[int(np.argmax(preds))]
                emotion_text = f"Emotion: {emotion_label}"

                cv2.putText(frame, emotion_label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              (255, 0, 0), 2)

                if tts_enabled and emotion_label != last_spoken_emotion:
                    speak_in_thread(emotion_label)
                    last_spoken_emotion = emotion_label
        else:
            emotion_text = "Emotion: No Face"
    except Exception:
        emotion_text = "Emotion: Error"

    # === ASL DETECTION ===
    try:
        hand_img = extract_hand(frame)
        hand_tensor = preprocess_asl_image(hand_img)
        asl_output = asl_model(hand_tensor)
        _, asl_pred = torch.max(asl_output.data, 1)
        asl_letter = label_binarizer.classes_[asl_pred]
        asl_text = f"ASL: {asl_letter}"

        if tts_enabled and asl_letter != last_spoken_asl:
            speak_in_thread(asl_letter)
            last_spoken_asl = asl_letter
    except Exception:
        asl_text = "ASL: Error"

    # === COMBINED DISPLAY ===
    combined_text = f"{asl_text} | {emotion_text}"
    cv2.putText(frame, combined_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    tts_status = "🗣️ ON" if tts_enabled else "🔇 OFF"
    cv2.putText(frame, f"TTS: {tts_status}", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 2)

    cv2.imshow('ASL + Emotion Recognition', frame)

    # === KEY HANDLING ===
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("👋 Quitting...")
        break
    elif key == ord('t'):
        if key_delay_counter == 0:
            tts_enabled = not tts_enabled
            print(f"TTS {'ENABLED 🗣️' if tts_enabled else 'DISABLED 🔇'}")
            key_delay_counter = 10  # Debounce counter
    if key_delay_counter > 0:
        key_delay_counter -= 1

cap.release()
cv2.destroyAllWindows()
