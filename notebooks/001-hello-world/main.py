"""Hello World with OpenVino"""

import cv2
import numpy as np
from openvino.runtime import Core

def classify():
    """Classifiy image with MobileNetV3 openvino runtime"""

    # load model
    ie = Core()
    model = ie.read_model(model="notebooks/001-hello-world/model/v3-small_224_1.0_float.xml")
    compiled_model = ie.compile_model(model, device_name="CPU")
    output_layer = compiled_model.output(0)

    # load image
    img = cv2.cvtColor(cv2.imread("notebooks/001-hello-world/data/coco.jpg"), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    input_img = np.expand_dims(img, axis=0)

    # inference
    result = compiled_model([input_img])[output_layer]
    result_idx = np.argmax(result)

    # print result
    imagenet_class = open("notebooks/001-hello-world/utils/imagenet_2012.txt").read().splitlines()
    imagenet_class = ["background"] + imagenet_class

    print("Result: ", imagenet_class[result_idx])

if __name__ == "__main__":

    classify()