'''
USAGE:
python preprocess_image.py --num-images 1200
'''

import os
import cv2
import random
import argparse
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num-images', default=1000, type=int,
                    help='Number of images to preprocess for each category')
args = vars(parser.parse_args())

print(f"Preprocessing {args['num_images']} from each category...")

root_path = '../input/asl_alphabet_train/asl_alphabet_train'
save_path = '../input/preprocessed_image'

dir_paths = sorted([d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))])

os.makedirs(save_path, exist_ok=True)

def process_category(dir_path):
    input_dir = os.path.join(root_path, dir_path)
    output_dir = os.path.join(save_path, dir_path)
    os.makedirs(output_dir, exist_ok=True)

    all_images = os.listdir(input_dir)
    processed = 0
    used_ids = set()

    while processed < args['num_images']:
        rand_id = random.randint(0, len(all_images) - 1)
        if rand_id in used_ids:
            continue

        used_ids.add(rand_id)
        img_path = os.path.join(input_dir, all_images[rand_id])
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (224, 224))

        out_path = os.path.join(output_dir, f"{dir_path}{processed}.jpg")
        cv2.imwrite(out_path, img)
        processed += 1

    return f"✅ {dir_path} done."

if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(executor.map(process_category, dir_paths), total=len(dir_paths)))

    for r in results:
        print(r)

    print('✅ DONE')
