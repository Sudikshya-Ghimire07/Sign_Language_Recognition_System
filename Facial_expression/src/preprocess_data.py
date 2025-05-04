import os
import numpy as np
import pandas as pd
from keras_preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
import pickle

IMAGE_SIZE = (48, 48)

def load_and_preprocess(data_dir):
    image_paths = []
    labels = []
    for label in os.listdir(data_dir):
        path = os.path.join(data_dir, label)
        for img_file in os.listdir(path):
            image_paths.append(os.path.join(path, img_file))
            labels.append(label)
        print(f"{label} completed")

    features = []
    for img_path in image_paths:
        img = load_img(img_path, color_mode='grayscale', target_size=IMAGE_SIZE)
        img_array = np.array(img)
        features.append(img_array)
    features = np.array(features).reshape(-1, 48, 48, 1) / 255.0

    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels)
    labels_categorical = pd.get_dummies(encoded_labels).values

    return features, labels_categorical, le

if __name__ == '__main__':
    os.makedirs("../data", exist_ok=True)

    for split in ['train', 'test']:
        print(f"\nProcessing {split} data...")
        features, labels, le = load_and_preprocess(f"../images/{split}")
        np.save(f"../data/{split}_data.npy", features)
        np.save(f"../data/{split}_labels.npy", labels)

    with open("../data/labels_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print("\n✅ All data processed and saved successfully.")
