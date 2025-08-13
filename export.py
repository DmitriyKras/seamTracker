import torch
from rknn.api import RKNN
from utils import SeamRegressor


INPUT_SHAPE = (1, 1, 640, 640)



model = SeamRegressor(1, 3)
# model.load_state_dict(torch.load('best.pt'))
model.eval()

# Export to ONNX
dummy_input = torch.ones(INPUT_SHAPE)
torch.onnx.export(
    model,
    dummy_input,
    'best.onnx',
    input_names=['input'],
    output_names=['output'],
    opset_version=17,
)

# Export to RKNN
rknn = RKNN()
rknn.config(mean_values=None, std_values=None, 
            target_platform='rk3588')
ret = rknn.load_onnx(model='best.onnx')
ret = rknn.build(do_quantization=False)
ret = rknn.export_rknn('best.rknn')
