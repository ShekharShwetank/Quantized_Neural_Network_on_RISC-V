# cam_capture_image.py
# This script captures an image from the webcam, preprocesses it, and saves it as a C array.

import cv2
import numpy as np
import tensorflow as tf
import os

# Configuration
# Quantization parameters from your mnist_model_params.c file
INPUT_SCALE = 0.00392157
INPUT_ZERO_POINT = -128
OUTPUT_C_FILE = "src/captured_image_data.h"

# Preprocessing Function
def preprocess_image(image_28x28_raw):
    image_float = image_28x28_raw.astype('float32') / 255.0
    rows = np.any(image_float > 0.1, axis=1)
    cols = np.any(image_float > 0.1, axis=0)

    if not np.any(rows):
        return np.full((12, 12), 0.0, dtype=np.float32)

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    buffer = 2
    ymin = max(0, ymin - buffer)
    ymax = min(28, ymax + buffer)
    xmin = max(0, xmin - buffer)
    xmax = min(28, xmax + buffer)

    cropped_image = image_float[ymin:ymax, xmin:xmax]
    resized_image = tf.image.resize(
        np.expand_dims(cropped_image, axis=-1), 
        (12, 12), 
        method='bilinear'
    ).numpy().squeeze()
    
    final_image_binary = (resized_image > 0.4).astype('float32')
    # cv2.imwrite('src/processed_frame.png', final_image_binary * 255) # Save the processed image for debugging and verification
    return final_image_binary

# Main script
def capture_and_save_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Capturing image from webcam. Press 'c' to capture.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        cv2.imshow('Webcam Feed (Press C to Capture)', frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.imwrite('src/captured_frame.png', frame)
            break

    # Convert to grayscale and resize to 28x28
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (28, 28), interpolation=cv2.INTER_AREA)

    # Preprocess and quantize the image
    preprocessed_image = preprocess_image(resized_frame)
    quantized_image = np.round(preprocessed_image / INPUT_SCALE + INPUT_ZERO_POINT).astype(np.int8)
    
    # Flatten and convert to a C array string
    hex_string = ", ".join([f"{val}" for val in quantized_image.flatten()])
    c_array_code = f"#ifndef CAPTURED_IMAGE_DATA_H\n#define CAPTURED_IMAGE_DATA_H\n#include <stdint.h>\n\nconst int8_t captured_image[144] = {{\n    {hex_string}\n}};\n\n#endif // CAPTURED_IMAGE_DATA_H"
    
    # Save the C array to a header file
    with open(OUTPUT_C_FILE, "w") as f:
        f.write(c_array_code)

    print(f"Captured image and saved as {OUTPUT_C_FILE}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_save_image()