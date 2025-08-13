import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Tuple
import cv2


class SeamDatasetSimple(Dataset):
    def __init__(self, root_dir: str):
        super().__init__()
        self.data = np.load(f"{root_dir}/images.npy", mmap_mode='r')
        # print(self.data.shape)
        self.n_frames = self.data.shape[0]
        self.gt = torch.from_numpy(np.loadtxt(f"{root_dir}/ground_truth.csv", delimiter=',').astype(int)[:, 1:] / np.array((720, 720))).float()

    def __len__(self) -> int:
        return self.n_frames
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        frames = self.data[idx].copy()
        # frames = cv2.resize(frames, (640, 640))
        # frames = cv2.blur(frames, (3,3), 0)
        frames = torch.from_numpy(frames).unsqueeze(0) / 255
        gt = self.gt[idx]
        return frames.float(), gt
    

class SeamDatasetStripe(Dataset):
    def __init__(self, root_dir: str):
        super().__init__()
        self.data = np.load(f"{root_dir}/images.npy", mmap_mode='r')
        # print(self.data.shape)
        self.n_frames = self.data.shape[0]
        self.gt = torch.from_numpy(np.loadtxt(f"{root_dir}/ground_truth.csv", delimiter=',').astype(int)[:, 1:])

    def __len__(self) -> int:
        return self.n_frames
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        frames = self.data[idx].copy()
        # frames = cv2.resize(frames, (640, 640))
        # frames = cv2.GaussianBlur(frames, (5,5), 0)
        frames = torch.from_numpy(frames).unsqueeze(0) / 255
        gt = self.gt[idx]
        x, y = gt  # coordinates of the center of stripe
        # print(x, y)
        frames = frames[:, :, x - 10 : x + 10]  # stripe only
        
        return frames.float(), torch.tensor(((0.5, y / 720))).float()
    

class SeamDataset1DSignal(Dataset):
    def __init__(self, root_dir: str):
        super().__init__()
        self.data = np.load(f"{root_dir}/images.npy", mmap_mode='r')
        # print(self.data.shape)
        self.n_frames = self.data.shape[0]
        self.gt = torch.from_numpy(np.loadtxt(f"{root_dir}/ground_truth.csv", delimiter=',').astype(int)[:, 1:])

    def __len__(self) -> int:
        return self.n_frames
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        frames = self.data[idx].copy()
        # frames = cv2.resize(frames, (640, 640))
        # frames = cv2.blur(frames, (5,5))
        frames = torch.from_numpy(frames).unsqueeze(0) / 255
        gt = self.gt[idx]
        x, y = gt  # coordinates of the center of stripe
        # print(x, y)
        frames = frames[:, :, x - 10 : x + 10].mean(dim=-1)  # stripe only
        
        return frames.float(), torch.tensor(((y / 720,))).float()


class SeamDatasetYOLO(Dataset):
    def __init__(self, root_dir: str):
        super().__init__()
        self.data = np.load(f"{root_dir}/images.npy", mmap_mode='r')
        # print(self.data.shape)
        self.n_frames = self.data.shape[0]
        self.gt = torch.from_numpy(np.loadtxt(f"{root_dir}/ground_truth.csv", delimiter=',').astype(int)[:, 1:] / np.array((720, 720))).float()

    def __len__(self) -> int:
        return self.n_frames
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        frames = self.data[idx].copy()
        frames = cv2.resize(frames, (640, 640))
        frames = cv2.cvtColor(frames, cv2.COLOR_GRAY2RGB)
        # frames = cv2.blur(frames, (3,3), 0)
        frames = torch.from_numpy(frames).permute(2, 0, 1) / 255
        gt = self.gt[idx]
        return frames.float(), gt


class SeamDatasetMask(Dataset):
    def __init__(self, root_dir: str):
        super().__init__()
        self.data = np.load(f"{root_dir}/images.npy", mmap_mode='r')
        # print(self.data.shape)
        self.n_frames = self.data.shape[0]
        self.gt = torch.from_numpy(np.loadtxt(f"{root_dir}/ground_truth.csv", delimiter=',').astype(int)[:, 1:] / np.array((720, 720))).float()

    def __len__(self) -> int:
        return self.n_frames - 1
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        frame = self.data[idx + 1].copy()
        frame_prev = self.data[idx].copy()
        # frames = cv2.resize(frames, (640, 640))
        # frames = cv2.blur(frames, (3,3), 0)
        frame = torch.from_numpy(frame) / 255
        frame_prev = torch.from_numpy(frame_prev) / 255
        gt = self.gt[idx]
        return torch.stack((frame, frame - frame_prev), dim=0).float(), gt
    

class SeamDatasetSegmentation(Dataset):
    def __init__(self, root_dir: str):
        super().__init__()
        self.data = np.load(f"{root_dir}/images.npy", mmap_mode='r')
        # print(self.data.shape)
        self.n_frames = self.data.shape[0]
        self.gt = torch.from_numpy(np.loadtxt(f"{root_dir}/ground_truth.csv", delimiter=',').astype(int)[:, 1:] / np.array((720, 720))).float()

    def __len__(self) -> int:
        return self.n_frames
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        frames = self.data[idx].copy()
        frames = cv2.resize(frames, (640, 640))
        # frames = cv2.blur(frames, (3,3), 0)
        frames = torch.from_numpy(frames).unsqueeze(0) / 255
        mask = torch.zeros_like(frames.squeeze())
        gt = self.gt[idx]
        x, y = (gt * 640).int()
        if 1 < x < 640 - 1 and 1 < y < 640 - 1:
            mask[y - 1 : y + 1, x - 1 : x + 1] = 1
        return frames.float(), mask, gt
