"""OpenVino API"""

import cv2
import numpy as np
from openvino.runtime import Core


def segmentation():
    """Segemetation"""

    ie = Core()
    net = ie.read_model(model="notebooks/002-openvino-api/model/segmentation.xml")
    compiled_net = ie.compile_model(net, "CPU")
    output_layer = compiled_net.output(0)

    image = cv2.imread("notebooks/002-openvino-api/data/coco_hollywood.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))
    image = np.expand_dims(image.transpose((2, 0, 1)), axis=0)

    # inference
    result = compiled_net([image])[output_layer]
    result = np.argmax(result)
    print(result)

if __name__ == "__main__":
    segmentation()



