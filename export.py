import torch
from rknn.api import RKNN
from utils import SeamRegressor, SeamRegressorTIMM
from pathlib import Path


INPUT_SHAPE = (1, 1, 1536, 640)
WEIGHTS = 'timm_mobilenetv4_conv_small.e1200_r224_in1k_640x1536_0.0029.pt'


model = SeamRegressorTIMM(1, 3)
model.load_state_dict(torch.load(WEIGHTS, map_location='cpu'))
model.eval()

# Export to ONNX
dummy_input = torch.ones(INPUT_SHAPE)
torch.onnx.export(
    model,
    dummy_input,
    f"{Path(WEIGHTS).stem}.onnx",
    input_names=['input'],
    output_names=['output'],
    opset_version=17,
)

# Export to RKNN
rknn = RKNN()
rknn.config(mean_values=None, std_values=None, 
            target_platform='rk3588')
ret = rknn.load_onnx(model=f"{Path(WEIGHTS).stem}.onnx")
ret = rknn.build(do_quantization=False)
ret = rknn.export_rknn(f"{Path(WEIGHTS).stem}.rknn")
