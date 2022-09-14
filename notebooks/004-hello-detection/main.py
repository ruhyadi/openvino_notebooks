"""Detection with OpenVINO"""

from openvino.runtime import Core
import cv2
import numpy as np

def detect(model_path: str, image_path: str):
    """Detect text with OpenVINO Runtime"""

    # load model
    ie = Core()
    model = ie.read_model(model_path)
    compiled_model = ie.compile_model(model, "CPU")
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output("boxes")

    # load image
    N, C, H, W = input_layer.shape
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (W, H))
    input_image = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), axis=0)

    # inference
    result = compiled_model([input_image])[output_layer]
    result = result[~np.all(result == 0, axis=1)]

    return image, resized_image, result

def convert_result_to_image(bgr_image, resized_image, boxes, threshold=0.3, conf_labels=True):
    # Define colors for boxes and descriptions.
    colors = {"red": (255, 0, 0), "green": (0, 255, 0)}

    # Fetch the image shapes to calculate a ratio.
    (real_y, real_x), (resized_y, resized_x) = bgr_image.shape[:2], resized_image.shape[:2]
    ratio_x, ratio_y = real_x / resized_x, real_y / resized_y

    # Convert the base image from BGR to RGB format.
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # Iterate through non-zero boxes.
    for box in boxes:
        # Pick a confidence factor from the last place in an array.
        conf = box[-1]
        if conf > threshold:
            # Convert float to int and multiply corner position of each box by x and y ratio.
            # If the bounding box is found at the top of the image, 
            # position the upper box bar little lower to make it visible on the image. 
            (x_min, y_min, x_max, y_max) = [
                int(max(corner_position * ratio_y, 10)) if idx % 2 
                else int(corner_position * ratio_x)
                for idx, corner_position in enumerate(box[:-1])
            ]

            # Draw a box based on the position, parameters in rectangle function are: image, start_point, end_point, color, thickness.
            rgb_image = cv2.rectangle(rgb_image, (x_min, y_min), (x_max, y_max), colors["green"], 3)

            # Add text to the image based on position and confidence.
            # Parameters in text function are: image, text, bottom-left_corner_textfield, font, font_scale, color, thickness, line_type.
            if conf_labels:
                rgb_image = cv2.putText(
                    rgb_image,
                    f"{conf:.2f}",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    colors["red"],
                    1,
                    cv2.LINE_AA,
                )

    return rgb_image

if __name__ == "__main__":
    model_path = "notebooks/004-hello-detection/model/horizontal-text-detection-0001.xml"
    image_path = "notebooks/004-hello-detection/data/intel_rnb.jpg"
    image, resized_image, result = detect(model_path, image_path)
    rgb_image = convert_result_to_image(image, resized_image, result)

    # write result
    save_path = "notebooks/004-hello-detection/data/result.jpg"
    cv2.imwrite(save_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Result saved to {save_path}")