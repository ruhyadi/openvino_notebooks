"""Segmentation with OpenVINO"""

import numpy as np
import cv2
from openvino.runtime import Core

def road_segmentation(model_path: str, image_path: str) -> np.ndarray:
    """Road segmentation with OpenVINO"""
    # load model
    ie = Core()
    model = ie.read_model(model=model_path)
    compiled_model = ie.compile_model(model=model, device_name="CPU")
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    N, C, H, W = input_layer.shape

    # load image
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (W, H))
    input_data = np.expand_dims(image.transpose(2, 0, 1), axis=0)

    # inference
    result = compiled_model([input_data])[output_layer]
    mask = np.argmax(result, axis=1)

    return result, mask

def visualization(mask, path):
    """Visualization"""
    img = mask.transpose(1, 2, 0)
    cv2.imwrite(path, img) # fail to save image
    print("Saved to", path)

if __name__ == "__main__":

    model_path = "notebooks/003-hello-segmentation/model/road-segmentation-adas-0001.xml"
    image_path = "notebooks/003-hello-segmentation/data/empty_road_mapillary.jpg"
    result, mask = road_segmentation(model_path, image_path)
    visualization(mask, "notebooks/003-hello-segmentation/data/mask.jpg")
