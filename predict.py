import torch
import cv2
import numpy as np


from utils import SeamRegressorTIMM
from utils import SeamDatasetSimple


ds = SeamDatasetSimple('new_data/2')
img, pt = ds[0]

# img = cv2.imread('new_data/7/0/0005.png', 0)
# img = cv2.resize(img, (320, 768))
model = SeamRegressorTIMM(in_channels=1, out_channels=3)
model.load_state_dict(torch.load('new_lcnet_100.ra2_in1k_0.0264_320x768.pt', map_location='cpu'))

# inp = torch.from_numpy(img).unsqueeze(0).unsqueeze(0) / 255

with torch.no_grad():
    x, y, _ = model(img.unsqueeze(0)).squeeze().numpy()

print(x, y)


img = (img.numpy().squeeze() * 255).astype(np.uint8)
H, W = img.shape
pt = pt.numpy() * np.array((W, H))
print(pt.astype(int).tolist())
img = cv2.circle(img, pt.astype(int).tolist(), 4, (255,), 1)
img = cv2.circle(img, (int(W * x), int(H * y)), 4, (125), 1)

cv2.imwrite('test.png', img)