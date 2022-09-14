"""Convert YOLOv5 model to ONNX and to OpenVINO IR"""

import sys
import torch
import onnx
import os
import subprocess

from pathlib import Path

sys.path.append("/raid/didir/Repository/openvino_notebooks/notebooks/102-pytorch-onnx-to-openvino/yolov5")
from models.common import AutoShape, DetectMultiBackend

def to_onnx(model_path: str):
    """Convert model to onnx"""
    device = torch.device("cpu")
    onnx_path = Path(model_path).with_suffix(".onnx")

    # load model
    model = DetectMultiBackend(model_path, device=device)
    model = AutoShape(model)

    # input data
    img = torch.zeros(1, 3, 640, 640).to(device)

    # convert onnx
    torch.onnx.export(
        model.cpu(),
        img.cpu(),
        onnx_path,
        input_names=['images'],
        output_names=['output'],
    )

    # check model
    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)

    # metadata
    d = {'stride': int(model.stride), 'names': model.names}
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(model_onnx, onnx_path)

    print(f"[INFO] Success convert model to ONNX saved to {onnx_path}")

def to_openvino(onnx_path: str):
    """Convert model from onnx to openvino"""
    onnx_path = Path(onnx_path)
    mo_command = f"""mo
                 --input_model "{str(onnx_path)}"
                 --data_type FP32
                 --output_dir "{onnx_path.parent}"
                 """
    mo_command = " ".join(mo_command.split())
    subprocess.run(mo_command.split(), check=True, env=os.environ)

    print(f"[INFO] Success convert model to OpenVINO IR saved to {onnx_path}")

if __name__ == "__main__":

    # model_path = "notebooks/102-pytorch-onnx-to-openvino/models/yolov5n.pt"
    # to_onnx(model_path=model_path)

    onnx_path = "/raid/didir/Repository/openvino_notebooks/notebooks/102-pytorch-onnx-to-openvino/models/yolov5n.onnx"
    to_openvino(onnx_path=onnx_path)