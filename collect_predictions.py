import numpy as np
import torch.nn.functional as F
import torch
from tqdm import tqdm
from utils import SeamDatasetSimple, SeamRegressor, SeamRegressorTIMM
import os
from torch.utils.data import DataLoader


dirs = [f"new_data/{d}" for d in os.listdir('new_data')]

model = SeamRegressorTIMM(1, 3)
model.load_state_dict(torch.load('best_score.pt'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


for d in dirs:
    ds = SeamDatasetSimple(d, (160, 1536))
    dl = DataLoader(ds, batch_size=64, shuffle=False)
    total_preds = []
    total_true = []

    with tqdm(dl, unit='batch') as tepoch:
        for i, (X_val, y_val) in enumerate(tepoch):
            tepoch.set_description(f"{d} VALIDATING")
            X_val, y_val = X_val.to(device), y_val.to(device)  # get data
            with torch.no_grad():
                y_pred = model(X_val)[:, :2]  # get predictions
            
            total_preds.append(y_pred.cpu().numpy())
            total_true.append(y_val.cpu().numpy())

    total_preds = np.concatenate(total_preds)
    total_true = np.concatenate(total_true)
    print(total_preds.shape)
    print(total_true.shape)
    np.save(f"{d}/preds.npy", total_preds)
    np.save(f"{d}/true.npy", total_true)