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
from utils import SeamDataset3D, SeamRegressorTIMM, resnet3d


# timm/efficientvit_m4.r224_in1k, linear, 0.0346, RMSE Loss + cls loss, 224x224, 128
# timm/efficientvit_m0.r224_in1k, linear, 0.0622, RMSE Loss + cls loss, 224x224, 128
# timm/tinynet_e.in1k, linear, 0.0270, RMSE Loss + cls loss, 224x224, 128
# timm/tinynet_e.in1k, linear, 0.0133, RMSE Loss + cls loss, 160x160, 128
# timm/lcnet_100.ra2_in1k, linear, 0.0105, RMSE Loss + cls loss, 160x160, 128
# timm/levit_128s.fb_dist_in1k, linear, 0.1235, RMSE Loss + cls loss, 224x224, 128

### NEW ANGLE ###
# timm/mobilenetv4_conv_small.e1200_r224_in1k, huber loss Best val RMSE: 0.0139 VAL MAE DX: 0.0033 VAL MAE DY: 0.0130  (160, 160)
# timm/mobilenetv4_conv_small.e1200_r224_in1k, huber loss BEST VAL RMSE: 0.0190 VAL MAE DX: 0.0030 VAL MAE DY: 0.0182  (160, 384)
# timm/mobilenetv4_conv_small.e1200_r224_in1k, huber loss Best val RMSE: 0.0182 VAL MAE DX: 0.0038 VAL MAE DY: 0.0172  (320, 768)
# timm/mobilenetv4_conv_small.e1200_r224_in1k, huber loss Best val RMSE: 0.0139 VAL MAE DX: 0.0033 VAL MAE DY: 0.0130  (640, 1536)


def build_dss(data_dir: str, input_shape) -> Tuple[ConcatDataset, ConcatDataset]:
    dirs = [f"{data_dir}/{d}" for d in os.listdir(data_dir)]
    # train, val = train_test_split(dirs, test_size=0.2, random_state=42)
    val = [f"{data_dir}/{d}" for d in os.listdir(data_dir) if d in ('3', '5', '6')]
    train = [d for d in dirs if d not in val]
    # train = [d for d in dirs]
    print(train, val)
    train = ConcatDataset([
        SeamDataset3D(d, input_shape) for d in train
    ])
    val = ConcatDataset([
        SeamDataset3D(d, input_shape) for d in val
    ])
    return train, val


if __name__ == '__main__':
    EPOCHS = 100
    BS = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    model = SeamRegressorTIMM(in_channels=4, out_channels=3)
    # model = resnet3d('resnet18')
    model.to(device)
    train_ds, val_ds = build_dss('new_data', (160, 384))
    
    # import cv2
    # img, pt = val_ds[1110]
    # img = (img.numpy().squeeze() * 255).astype(np.uint8)
    # pt = pt.numpy() * 720
    # img = cv2.circle(img, pt.astype(int).tolist(), 5, (255,), 2)
    # cv2.imwrite('test.png', img)
    train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BS, shuffle=False)
    train_loss = nn.HuberLoss()
    train_loss_clf = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=0.0001)
    best_rmse = 1
    scaler = torch.amp.GradScaler()

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
                with torch.autocast('cuda', torch.float16):
                    y_pred = model(X_train)  # get predictions
                    y_train_cls = torch.all(y_train < 1, dim=-1)

                    # loss_reg = torch.sqrt(train_loss(y_pred[:, :2], y_train))
                    loss_reg = train_loss(y_pred[:, :2], y_train)
                    loss_clf = train_loss_clf(y_pred[:, 2], y_train_cls.float())
                    loss = loss_reg + 0.1*loss_clf

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                total_loss += loss_reg.item()
                total_loss_clf += loss_clf.item()
                tepoch.set_postfix(loss=total_loss / (i + 1))
                time.sleep(0.1)
        
        total_loss = total_loss / (i + 1)
        total_loss_clf = total_loss_clf / (i + 1)
        model.eval()
        total_val_loss = 0
        total_preds = []
        total_true = []
        dx, dy = 0, 0
        with tqdm(val_dl, unit='batch') as tepoch:
            for i, (X_val, y_val) in enumerate(tepoch):
                tepoch.set_description(f"EPOCH {epoch + 1}/{EPOCHS} VALIDATING")
                X_val, y_val = X_val.to(device), y_val.to(device)  # get data
                with torch.autocast('cuda', torch.float16), torch.no_grad():
                    y_pred = model(X_val)  # get predictions
                    y_val_cls = torch.all(y_val < 1, dim=-1)
                    loss = torch.sqrt(torch.square(y_pred[:, :2] - y_val).sum(dim=-1)).mean()
                    
                total_preds.append(F.sigmoid(y_pred[:, 2]).cpu().numpy())
                total_true.append(y_val_cls.cpu().numpy())

                
                d = torch.abs(y_pred[:, :2] - y_val).mean(dim=0)
                dx += d[0].item()
                dy += d[1].item()
                # loss = torch.sqrt(val_loss(y_pred, y_val))
                total_val_loss += loss.item()
        total_val_loss /= (i + 1)
        dx /= (i + 1)
        dy /= (i + 1)

        if total_val_loss < best_rmse:
            best_rmse = total_val_loss
            torch.save(model.state_dict(), 'best_score.pt')

        print(f"TRAIN REG LOSS: {total_loss:.4f}")
        print(f"TRAIN CLF LOSS: {total_loss_clf:.4f}")
        print(f"VAL RMSE: {total_val_loss:.4f}")
        print(f"VAL MAE DX: {dx:.4f} VAL MAE DY: {dy:.4f}")
        total_preds = (np.concatenate(total_preds) > 0.5).astype(int)
        total_true = np.concatenate(total_true)
        pr, rec, f1, _ = precision_recall_fscore_support(total_true, total_preds, average='binary')
        print(f"F1-score: {f1}\nPrecision: {pr}\nRecall: {rec}")

    print(f"Best val RMSE: {best_rmse:.4f}")
