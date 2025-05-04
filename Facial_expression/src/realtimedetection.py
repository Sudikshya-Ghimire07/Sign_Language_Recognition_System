import cv2
import os
import sys
import numpy as np
import pyttsx3
import threading
import queue
import time
from keras.models import model_from_json, Sequential

# Enable TTS via command-line flag
enable_tts = "--tts" in sys.argv

# TTS setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)
tts_queue = queue.Queue()

def tts_worker():
    while True:
        emotion = tts_queue.get()
        if emotion is None:
            break
        engine.say(emotion)
        engine.runAndWait()
        tts_queue.task_done()

# Launch TTS background thread
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

if enable_tts:
    print("🗣️ Real-time translation is ENABLED.")
else:
    print("🔇 Real-time translation is DISABLED.")

# Paths
json_file_path = r"E:\Sign_Language_Detection\Facial_expression\facialemotionmodel.json"
weights_file_path = r"E:\Sign_Language_Detection\Facial_expression\facialemotionmodel.h5"

# Load model
if not os.path.exists(json_file_path) or not os.path.exists(weights_file_path):
    print("❌ Model files not found.")
    exit(1)

with open(json_file_path, "r") as f:
    model = model_from_json(f.read(), custom_objects={'Sequential': Sequential})
model.load_weights(weights_file_path)
print("✅ Model loaded.")

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Feature extraction
def extract_features(img):
    img = img.reshape(1, 48, 48, 1)
    return img / 255.0

# Webcam
cam = cv2.VideoCapture(0)
tts_runtime_toggle = enable_tts
last_emotion = ""
emotion_history = []
cooldown_time = 2  # seconds
last_spoken = time.time() - cooldown_time

while True:
    start_time = time.time()
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        processed = extract_features(np.array(face))

        pred = model.predict(processed, verbose=0)[0]
        label_idx = np.argmax(pred)
        emotion = labels[label_idx]
        confidence = pred[label_idx]

        color = (0, 255, 0) if confidence > 0.7 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        text = f"{emotion} ({confidence*100:.1f}%)"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # TTS only if emotion changes and cooldown passes
        if tts_runtime_toggle and emotion != last_emotion and (time.time() - last_spoken) > cooldown_time:
            last_emotion = emotion
            last_spoken = time.time()
            tts_queue.put(emotion)

        # Add to emotion history
        emotion_history.append(emotion)
        if len(emotion_history) > 5:
            emotion_history.pop(0)

    # Display FPS
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # TTS Status
    cv2.putText(frame, "🗣️ TTS ON" if tts_runtime_toggle else "🔇 TTS OFF",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0) if tts_runtime_toggle else (0, 0, 255), 2)

    # Emotion history display
    for i, emo in enumerate(emotion_history[::-1]):
        cv2.putText(frame, f"History: {emo}", (10, 90 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Facial Emotion Recognition", frame)

    # Key bindings
    key = cv2.waitKey(1) & 0xFF
    if key in [27, ord('q')]:  # ESC or q
        break
    elif key == ord('t'):
        tts_runtime_toggle = not tts_runtime_toggle
        print("🔄 TTS", "ENABLED" if tts_runtime_toggle else "DISABLED")
    elif key == ord('s'):
        filename = f"screenshot_{int(time.time())}.png"
        cv2.imwrite(filename, frame)
        print(f"📸 Saved frame to {filename}")
    elif key == ord('c'):
        emotion_history.clear()
        print("🧹 Emotion history cleared.")

# Cleanup
tts_queue.put(None)
tts_thread.join()
cam.release()
cv2.destroyAllWindows()
