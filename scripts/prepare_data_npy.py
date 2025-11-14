import numpy as np
import cv2
from glob import glob
import os
from tqdm import tqdm

DIR = 'new_data'
dirs = [f"{DIR}/{d}" for d in os.listdir(DIR)]
print(dirs)

for d in tqdm(dirs, total=len(dirs)):
    images = sorted(glob(f"{d}/0/*.png"))
    gt = np.loadtxt(f"{d}/ground_truth.csv", delimiter=',').astype(int)
    id_start = gt[0, 0]
    id_end = gt[-1, 0]
    images = images[id_start : id_end + 1]
    images = np.stack([cv2.imread(i, 0) for i in images])
    np.save(f"{d}/images.npy", images)