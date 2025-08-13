import numpy as np
from glob import glob
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pathlib import Path
import cv2

DIR = 'data'
TARGET_DIR = 'data_yolo'
dirs = [f"{DIR}/{d}" for d in os.listdir(DIR)]
print(dirs)

os.makedirs(f"{TARGET_DIR}/train/images", exist_ok=True)
os.makedirs(f"{TARGET_DIR}/train/labels", exist_ok=True)
os.makedirs(f"{TARGET_DIR}/val/images", exist_ok=True)
os.makedirs(f"{TARGET_DIR}/val/labels", exist_ok=True)

train_d, val_d = train_test_split(dirs, test_size=0.2, random_state=42)


for d in tqdm(train_d, total=len(train_d)):
    images = sorted(glob(f"{d}/0/*.pgm"))
    gt = np.loadtxt(f"{d}/ground_truth.csv", delimiter=',').astype(int)
    id_start = gt[0, 0]
    id_end = gt[-1, 0]
    images = images[id_start : id_end + 1]
    dir_n = Path(d).stem
    for pt, img in tqdm(zip(gt, images), total=len(images)):
        image = cv2.imread(img)
        img_name = Path(img).stem
        cv2.imwrite(f"{TARGET_DIR}/train/images/{dir_n}_{img_name}.jpg", image)
        x, y = pt[1:]
        label = f"0 {x / 720} 0.5 {20 / 720} 0.99 {x / 720} {y / 720} 2"
        with open(f"{TARGET_DIR}/train/labels/{dir_n}_{img_name}.txt", 'w') as f:
            f.write(label)


for d in tqdm(val_d, total=len(val_d)):
    images = sorted(glob(f"{d}/0/*.pgm"))
    gt = np.loadtxt(f"{d}/ground_truth.csv", delimiter=',').astype(int)
    id_start = gt[0, 0]
    id_end = gt[-1, 0]
    images = images[id_start : id_end + 1]
    dir_n = Path(d).stem
    for pt, img in tqdm(zip(gt, images), total=len(images)):
        image = cv2.imread(img)
        img_name = Path(img).stem
        cv2.imwrite(f"{TARGET_DIR}/val/images/{dir_n}_{img_name}.jpg", image)
        x, y = pt[1:]
        label = f"0 {x / 720} 0.5 {20 / 720} 0.99 {x / 720} {y / 720} 2"
        with open(f"{TARGET_DIR}/val/labels/{dir_n}_{img_name}.txt", 'w') as f:
            f.write(label)
