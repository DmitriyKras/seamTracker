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
from utils import SeamDatasetSegmentation, TinyUNet


# ShuffleNet, linear, 0.0113, SmoothL1Loss
# ShuffleNet, linear, 0.0096, RMSE Loss
# SqueezeNet, linear, 0.0769, RMSE Loss
# ShuffleNet, linear, 0.0052, RMSE Loss, without 9 and 10
# ShuffleNet, sigmoid, 0.0089, RMSE Loss, without 9 and 10
# ShuffleNet, sigmoid, 0.012, RMSE Loss + cls loss
# ShuffleNet, linear, 0.0121, RMSE Loss + cls loss


def batch_center_of_mass(weights: torch.Tensor):
    """
    Вычисляет центры масс для батча 2D масок.
    
    Аргументы:
        weights (torch.Tensor): Тензор размерности [B, H, W]
        
    Возвращает:
        torch.Tensor: Тензор размерности [B, 2] с координатами центров масс
    """
    device = weights.device
    b, h, w = weights.shape
    
    # Создаем сетку координат
    y_coords = torch.arange(h, device=device).float().view(1, h, 1)
    x_coords = torch.arange(w, device=device).float().view(1, 1, w)
    
    # Нормировочный коэффициент
    total_weight = weights.sum(dim=(1, 2), keepdim=True)
    
    # Вычисляем центры масс
    x_center = (weights * x_coords).sum(dim=(1, 2)) / total_weight.squeeze()
    y_center = (weights * y_coords).sum(dim=(1, 2)) / total_weight.squeeze()
    
    return torch.stack([x_center, y_center], dim=1)


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice



def build_dss(data_dir: str) -> Tuple[ConcatDataset, ConcatDataset]:
    dirs = [f"{data_dir}/{d}" for d in os.listdir(data_dir)]
    train, val = train_test_split(dirs, test_size=0.2, random_state=42)
    train = ConcatDataset([
        SeamDatasetSegmentation(d) for d in train
    ])
    val = ConcatDataset([
        SeamDatasetSegmentation(d) for d in val
    ])
    return train, val


if __name__ == '__main__':
    EPOCHS = 30
    BS = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TinyUNet(1, 1)
    model.to(device)
    train_ds, val_ds = build_dss('data')
    
    # import cv2
    # img, pt = val_ds[1110]
    # img = (img.numpy().squeeze() * 255).astype(np.uint8)
    # pt = pt.numpy() * 720
    # img = cv2.circle(img, pt.astype(int).tolist(), 5, (255,), 2)
    # cv2.imwrite('test.png', img)
    train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BS, shuffle=False)
    train_loss = DiceLoss()
    #train_loss = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=0.0001)
    best_rmse = 1


    for epoch in range(EPOCHS):
        model.train()  # set model to training mode
        total_loss = 0  # init total losses and metrics
        with tqdm(train_dl, unit='batch') as tepoch:
            for i, (X_train, y_train, coords) in enumerate(tepoch):
                tepoch.set_description(f"EPOCH {epoch + 1}/{EPOCHS} TRAINING")
                X_train, y_train = X_train.to(device), y_train.to(device)  # get data
                # print(X_train.size())
                y_pred = model(X_train).squeeze()  # get predictions
                loss = train_loss(y_pred, y_train)
                optimizer.zero_grad()
                loss.backward()  # back propogation
                optimizer.step()  # optimizer's step
                total_loss += loss.item()
                tepoch.set_postfix(loss=total_loss / (i + 1))
                time.sleep(0.1)
        
        total_loss = total_loss / (i + 1)
        if total_loss < best_rmse:
            best_rmse = total_loss
            # torch.save(model.state_dict(), 'data/dynamic_abs_best.pt')
        model.eval()
        total_val_loss = 0
        total_rmse = 0
        total_preds = []
        total_true = []
        with tqdm(val_dl, unit='batch') as tepoch:
            for i, (X_val, y_val, coords) in enumerate(tepoch):
                tepoch.set_description(f"EPOCH {epoch + 1}/{EPOCHS} VALIDATING")
                X_val, y_val = X_val.to(device), y_val.to(device)  # get data
                coords = coords.to(device)
                with torch.no_grad():
                    y_pred = model(X_val).squeeze()  # get predictions

                # total_preds.append(F.sigmoid(y_pred).cpu().numpy().flatten())
                # total_true.append(y_val.cpu().numpy().flatten())
                loss = train_loss(y_pred, y_val)
                rmse = torch.sqrt(torch.square(batch_center_of_mass(F.sigmoid(y_pred)) / y_pred.size(1) - coords).sum(dim=-1)).mean()
                total_rmse += rmse.item()
                total_val_loss += loss.item()
        total_val_loss /= (i + 1)
        total_rmse /= (i + 1)
        print(f"TRAIN LOSS: {total_loss:.4f}")
        print(f"VAL LOSS: {total_val_loss:.4f}")
        print(f"VAL RMSE: {total_rmse:.4f}")
        # total_preds = (np.concatenate(total_preds) > 0.5).astype(int)
        # total_true = np.concatenate(total_true)
        # f1, pr, rec, _ = precision_recall_fscore_support(total_true, total_preds, average='binary')
        # print(f"F1-score: {f1}\nPrecision: {pr}\nRecall: {rec}")
