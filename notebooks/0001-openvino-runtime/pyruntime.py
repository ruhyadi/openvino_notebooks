"""
OpenVino Runtime with Python
Source: https://docs.openvino.ai/latest/openvino_docs_OV_UG_Integrate_OV_with_your_application.html
"""

import openvino.runtime as ov
import numpy as np
import cv2

def inference():
    """Inference with OpenVINO runtime"""

    core = ov.Core()
    compiled_model = core.compile_model(
        "notebooks/0001-openvino-runtime/models/v3-small_224_1.0_float.xml", 
        "CPU"
    )
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    input_data = np.random.random((1, 224, 224, 3)).astype(np.float32)

    request = compiled_model.create_infer_request()
    request.infer(inputs={input_layer.any_name: np.random.random((1, 224, 224, 3)).astype(np.float32)})
    result = request.get_output_tensor(output_layer.index).data
    result_idx = np.argmax(result, axis=1)

    print(result_idx)

if __name__ == "__main__":
    inference()