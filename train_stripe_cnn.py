import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam
from typing import Tuple
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import time
from utils import SeamRegressor, SeamDatasetStripe


# ShuffleNet, linear, 0.0252, RMSE Loss
    

def build_dss(data_dir: str) -> Tuple[ConcatDataset, ConcatDataset]:
    dirs = [f"{data_dir}/{d}" for d in os.listdir(data_dir)]
    train, val = train_test_split(dirs, test_size=0.2, random_state=42)
    train = ConcatDataset([
        SeamDatasetStripe(d) for d in train
    ])
    val = ConcatDataset([
        SeamDatasetStripe(d) for d in val
    ])
    return train, val


if __name__ == '__main__':
    EPOCHS = 20
    BS = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SeamRegressor(1, 2)
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
    train_loss = nn.MSELoss()
    val_loss = nn.MSELoss()
    optimizer = Adam(model.parameters())
    best_rmse = 1


    for epoch in range(EPOCHS):
        model.train()  # set model to training mode
        total_loss = 0  # init total losses and metrics
        with tqdm(train_dl, unit='batch') as tepoch:
            for i, (X_train, y_train) in enumerate(tepoch):
                tepoch.set_description(f"EPOCH {epoch + 1}/{EPOCHS} TRAINING")
                X_train, y_train = X_train.to(device), y_train.to(device)  # get data
                # print(X_train.size())
                # print(y_train.size())
                y_pred = model(X_train)  # get predictions
                loss = torch.sqrt(train_loss(y_pred, y_train))
                optimizer.zero_grad()
                loss.backward()  # back propogation
                optimizer.step()  # optimizer's step
                total_loss += torch.sqrt(loss).item()
                tepoch.set_postfix(loss=total_loss / (i + 1))
                time.sleep(0.1)
        
        total_loss = total_loss / (i + 1)
        if total_loss < best_rmse:
            best_rmse = total_loss
            # torch.save(model.state_dict(), 'data/dynamic_abs_best.pt')
        model.eval()
        total_val_loss = 0
        with tqdm(val_dl, unit='batch') as tepoch:
            for i, (X_val, y_val) in enumerate(tepoch):
                tepoch.set_description(f"EPOCH {epoch + 1}/{EPOCHS} VALIDATING")
                X_val, y_val = X_val.to(device), y_val.to(device)  # get data
                with torch.no_grad():
                    y_pred = model(X_val)  # get predictions
                loss = torch.sqrt(val_loss(y_pred, y_val))
                total_val_loss += loss.item()
        total_val_loss /= (i + 1)
        print(f"TRAIN LOSS: {total_loss:.4f}\n")
        print(f"VAL RMSE: {total_val_loss:.4f}\n")
