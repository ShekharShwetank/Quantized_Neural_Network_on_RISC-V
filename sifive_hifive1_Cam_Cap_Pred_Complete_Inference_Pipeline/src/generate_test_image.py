import numpy as np
import tensorflow as tf
from skimage.transform import resize
import io

def generate_c_hex_array(digit, INPUT_SCALE, INPUT_ZERO_POINT):
    """
    Generates a synthetic 28x28 image of a digit, preprocesses it,
    and returns its C-style quantized hex array.
    """
    # Create a 28x28 image with a white digit on a black background
    image_28x28_raw = np.zeros((28, 28), dtype=np.uint8)
    
    # Simple logic to "draw" a digit. A more complex method may be needed for other digits.
    if digit == 3:
        # Example drawing a '3'
        image_28x28_raw[4:6, 10:18] = 255
        image_28x28_raw[6:10, 16:18] = 255
        image_28x28_raw[12:14, 10:18] = 255
        image_28x28_raw[14:18, 16:18] = 255
        image_28x28_raw[20:22, 10:18] = 255
    else:
        # Fallback for other digits, prints a simple square to avoid error
        image_28x28_raw[10:18, 10:18] = 255

    # Normalize the image
    image_28x28 = image_28x28_raw.astype('float32') / 255.0

    # Find bounding box
    rows = np.any(image_28x28 > 0.05, axis=1) # Use a very low threshold
    cols = np.any(image_28x28 > 0.05, axis=0)

    # Check if any non-zero pixels were found
    if not np.any(rows):
        # Return an array of all zeroes if no digit is found
        quantized_image = np.full((144,), INPUT_ZERO_POINT, dtype=np.int8)
    else:
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        
        buffer = 2
        ymin = max(0, ymin - buffer)
        ymax = min(28, ymax + buffer)
        xmin = max(0, xmin - buffer)
        xmax = min(28, xmax + buffer)

        cropped_image = image_28x28[ymin:ymax, xmin:xmax]

        resized_image = tf.image.resize(
            np.expand_dims(cropped_image, axis=-1), 
            (12, 12), 
            method='bilinear'
        ).numpy().squeeze()

        final_image_binary = (resized_image > 0.4).astype('float32')
        
        # Quantize the image to int8
        quantized_image = np.round(final_image_binary / INPUT_SCALE + INPUT_ZERO_POINT)
        quantized_image = np.clip(quantized_image, -128, 127).astype(np.int8)

    # Convert the flattened array to a C-style hex string
    hex_string = ", ".join([f"{val}" for val in quantized_image.flatten()])
    
    return f"const int8_t input_data_new[144] = {{\n    {hex_string}\n}};"

if __name__ == "__main__":
    # Using the quantization values from your mnist_model_params.c
    INPUT_SCALE = 0.00392157
    INPUT_ZERO_POINT = -128
    
    # Generate C array for a new digit (e.g., the digit 3)
    c_array_code = generate_c_hex_array(3, INPUT_SCALE, INPUT_ZERO_POINT)
    
    print("--------------------------------------------------")
    print("Copy this code into your main.c file:")
    print("--------------------------------------------------")
    print(c_array_code)
    print("--------------------------------------------------")