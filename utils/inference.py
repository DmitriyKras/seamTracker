import numpy as np
import cv2
from rknn.api import RKNN


def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))


class SeamRegressorRKNN:
    def __init__(self, model_path: str, input_shape=(640, 640)):
        self.model = RKNN()
        self.model.load_rknn(model_path)
        self.model.init_runtime(target='rk3588')
        self.input_shape = input_shape

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        img = cv2.resize(img, self.input_shape) / 255
        img = img.reshape((1, img.shape[-2], img.shape[-1]))
        img = img.astype(np.float16)
        return img
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = self.preprocess(img)
        outputs = self.model.inference(inputs=[img])
        outputs = sigmoid(outputs).squeeze()
        return outputs  # x, y, score
