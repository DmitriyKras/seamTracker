import torch
from torch.utils.data import DataLoader, ConcatDataset
from math import floor
from typing import Tuple
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from ultralytics import YOLO
from utils import SeamDatasetYOLO
import os

# yolov11n-pose 0.0475 RMSE


def build_dss(data_dir: str) -> Tuple[ConcatDataset, ConcatDataset]:
    dirs = [f"{data_dir}/{d}" for d in os.listdir(data_dir)]
    train, val = train_test_split(dirs, test_size=0.2, random_state=42)
    train = ConcatDataset([
        SeamDatasetYOLO(d) for d in train
    ])
    val = ConcatDataset([
        SeamDatasetYOLO(d) for d in val
    ])
    return train, val


if __name__ == '__main__':
    BS = 16
    yolo = YOLO('runs/pose/train2/weights/best.pt')
    train_ds, val_ds = build_dss('data')
    val_dl = DataLoader(val_ds, batch_size=BS, shuffle=False)
    total_val_loss = 0
    with tqdm(val_dl, unit='batch') as tepoch:
        for i, (X_val, y_val) in enumerate(tepoch):
            tepoch.set_description(f"VALIDATING")
            y_val = y_val.cuda()
            out = yolo(X_val, conf=0.4, verbose=False)
            y_pred = torch.stack([pr[0].keypoints.xyn.squeeze() for pr in out], dim=0)
            loss = torch.sqrt(torch.square(y_pred - y_val).sum(dim=-1)).mean()
            # loss = torch.sqrt(val_loss(y_pred, y_val))
            total_val_loss += loss.item()

    total_val_loss /= (i + 1)
    print(f"VAL RMSE: {total_val_loss:.4f}\n")
