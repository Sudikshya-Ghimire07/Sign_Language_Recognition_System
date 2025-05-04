import os
import pandas as pd

def create_csv(data_dir, output_file):
    image_paths = []
    labels = []
    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        for img in os.listdir(class_dir):
            image_paths.append(os.path.join(class_dir, img))
            labels.append(label)
        print(f"{label} done")
    df = pd.DataFrame({"image": image_paths, "label": labels})
    df.to_csv(output_file, index=False)
    print(f"Saved CSV to {output_file}")

if __name__ == '__main__':
    create_csv("../images/train", "../data/train.csv")
    create_csv("../images/test", "../data/test.csv")
