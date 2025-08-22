from utils import SeamRegressorRKNN
import time
import numpy as np


N_RUNS = 100


model = SeamRegressorRKNN('best_mbnet3.rknn', (160, 160))
start = time.time()

for _ in range(N_RUNS):
    res = model(np.ones((160, 160)))

end = time.time()
print(f"Output shape {res.shape}")
print(f"Inference speed {(end - start) / N_RUNS * 1000:.2f} ms")