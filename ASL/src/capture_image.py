import os
import cv2
import time
import string

# Define save path
DATA_DIR = r'E:\Sign_Language_Recognition_System\ASL\input\asl_alphabet_train\asl_alphabet_train'
dataset_size = 50

# ROI (box) position
x1, y1, box_size = 100, 100, 200
x2, y2 = x1 + box_size, y1 + box_size

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Choose which folder/class to capture
for folder_name in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']:  # Add more letters if needed
    class_dir = os.path.join(DATA_DIR, folder_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    # Count how many images already exist
    existing_images = len(os.listdir(class_dir))
    counter = existing_images

    print(f'Capturing for class "{folder_name}" (starting at image {counter})')
    print('Press any alphabet key to start...')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading from camera.")
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f'Press any alphabet key to capture "{folder_name}"', 
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(1) & 0xFF
        if chr(key).upper() in string.ascii_uppercase:
            # Countdown 5 to 1
            for i in range(5, 0, -1):
                ret, countdown_frame = cap.read()
                if not ret:
                    continue
                cv2.rectangle(countdown_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(countdown_frame, f'Starting in {i}', (180, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.imshow('frame', countdown_frame)
                cv2.waitKey(1000)
            break

    while counter < existing_images + dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Skipped a bad frame.")
            continue

        roi = frame[y1:y2, x1:x2]
        img_path = os.path.join(class_dir, f'{counter}.jpg')
        try:
            cv2.imwrite(img_path, roi)
            print(f"Saved: {img_path}")
        except Exception as e:
            print(f"Failed to save {img_path}: {e}")
            continue

        cv2.imshow('Hand ROI', roi)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Capture interrupted by user.")
            break

        counter += 1

cap.release()
cv2.destroyAllWindows()
