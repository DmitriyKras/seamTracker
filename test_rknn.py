from utils import SeamRegressorRKNN
import time
import numpy as np


N_RUNS = 10


model = SeamRegressorRKNN('best.rknn')
start = time.time()

for _ in range(N_RUNS):
    res = model(np.ones((640, 640)))

end = time.time()
print(f"Output shape {res.shape}")
print(f"Inference speed {(end - start) * 1000:.2f} ms")