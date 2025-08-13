import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam
import numpy as np
import torch.nn.functional as F
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import os
import time
from utils import SeamDatasetMask, SeamRegressor


# ShuffleNet, linear, 0.0113, SmoothL1Loss
# ShuffleNet, linear, 0.0096, RMSE Loss
# SqueezeNet, linear, 0.0769, RMSE Loss
# ShuffleNet, linear, 0.0052, RMSE Loss, without 9 and 10
# ShuffleNet, sigmoid, 0.0089, RMSE Loss, without 9 and 10
# ShuffleNet, sigmoid, 0.012, RMSE Loss + cls loss
# ShuffleNet, linear, 0.0121, RMSE Loss + cls loss



    

def build_dss(data_dir: str) -> Tuple[ConcatDataset, ConcatDataset]:
    dirs = [f"{data_dir}/{d}" for d in os.listdir(data_dir)]
    train, val = train_test_split(dirs, test_size=0.2, random_state=42)
    train = ConcatDataset([
        SeamDatasetMask(d) for d in train
    ])
    val = ConcatDataset([
        SeamDatasetMask(d) for d in val
    ])
    return train, val


if __name__ == '__main__':
    EPOCHS = 30
    BS = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SeamRegressor(in_channels=2, out_channels=3)
    model.to(device)
    train_ds, val_ds = build_dss('data')
    
    import cv2
    img, pt = val_ds[200]
    img = (img.numpy().squeeze() * 255).astype(np.uint8)[1]
    pt = pt.numpy() * 720
    img = cv2.circle(img, pt.astype(int).tolist(), 5, (255,), 2)
    cv2.imwrite('test.png', img)
    train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BS, shuffle=False)
    train_loss = nn.MSELoss()
    train_loss_clf = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters())
    best_rmse = 1


    for epoch in range(EPOCHS):
        model.train()  # set model to training mode
        total_loss = 0  # init total losses and metrics
        total_loss_clf = 0
        with tqdm(train_dl, unit='batch') as tepoch:
            for i, (X_train, y_train) in enumerate(tepoch):
                tepoch.set_description(f"EPOCH {epoch + 1}/{EPOCHS} TRAINING")
                X_train, y_train = X_train.to(device), y_train.to(device)  # get data
                # print(X_train.size())
                # print(y_train.size())
                y_pred = model(X_train)  # get predictions
                y_train_cls = torch.any(y_train > 1, dim=-1)

                loss_reg = torch.sqrt(train_loss(y_pred[:, :2], y_train))
                loss_clf = train_loss_clf(y_pred[:, 2], y_train_cls.float())
                loss = loss_reg + 0.5*loss_clf
                optimizer.zero_grad()
                loss.backward()  # back propogation
                optimizer.step()  # optimizer's step
                total_loss += loss_reg.item()
                total_loss_clf += loss_clf.item()
                tepoch.set_postfix(loss=total_loss / (i + 1))
                time.sleep(0.1)
        
        total_loss = total_loss / (i + 1)
        total_loss_clf = total_loss_clf / (i + 1)
        if total_loss < best_rmse:
            best_rmse = total_loss
            # torch.save(model.state_dict(), 'data/dynamic_abs_best.pt')
        model.eval()
        total_val_loss = 0
        total_preds = []
        total_true = []
        with tqdm(val_dl, unit='batch') as tepoch:
            for i, (X_val, y_val) in enumerate(tepoch):
                tepoch.set_description(f"EPOCH {epoch + 1}/{EPOCHS} VALIDATING")
                X_val, y_val = X_val.to(device), y_val.to(device)  # get data
                with torch.no_grad():
                    y_pred = model(X_val)  # get predictions

                y_val_cls = torch.any(y_val > 1, dim=-1)
                total_preds.append(F.sigmoid(y_pred[:, 2]).cpu().numpy())
                total_true.append(y_val_cls.cpu().numpy())

                loss = torch.sqrt(torch.square(y_pred[:, :2] - y_val).sum(dim=-1)).mean()
                # loss = torch.sqrt(val_loss(y_pred, y_val))
                total_val_loss += loss.item()
        total_val_loss /= (i + 1)
        print(f"TRAIN REG LOSS: {total_loss:.4f}")
        print(f"TRAIN CLF LOSS: {total_loss_clf:.4f}")
        print(f"VAL RMSE: {total_val_loss:.4f}")
        total_preds = (np.concatenate(total_preds) > 0.5).astype(int)
        total_true = np.concatenate(total_true)
        f1, pr, rec, _ = precision_recall_fscore_support(total_true, total_preds, average='binary')
        print(f"F1-score: {f1}\nPrecision: {pr}\nRecall: {rec}")
