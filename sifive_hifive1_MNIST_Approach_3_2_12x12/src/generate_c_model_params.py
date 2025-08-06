'''
    generate_c_model_params.py
    Author: Shwetank Shekhar
'''
import tensorflow as tf
import numpy as np
import os

TFLITE_MODEL_PATH = "src/mnist_model_int8.tflite"
C_HEADER_FILE = "src/mnist_model_params.h"
C_SOURCE_FILE = "src/mnist_model_params.c"

def quantize_input(image_data_float32, scale, zero_point):
    quantized_data = np.round(image_data_float32 / scale + zero_point)
    quantized_data = np.clip(quantized_data, -128, 127).astype(np.int8)
    return quantized_data

def generate_c_arrays_from_tflite(model_path, header_file, source_file):
    print(f"Loading TFLite model from: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    tensor_details = interpreter.get_tensor_details()
    ops_details_raw = interpreter._get_ops_details()

    with open(header_file, "w") as hf, open(source_file, "w") as sf:
        hf.write("#ifndef MNIST_MODEL_PARAMS_H\n#define MNIST_MODEL_PARAMS_H\n\n#include <stdint.h>\n\n")
        hf.write("#define MAX_N_ACTIVATIONS 144\n#define ACTIVATION_BITS 8\n\n")
        
        sf.write("#include \"mnist_model_params.h\"\n\n")

        layer_idx = 1
        for op_detail in ops_details_raw:
            if op_detail.get('op_name') == 'FULLY_CONNECTED':
                print(f"Processing FULLY_CONNECTED layer {layer_idx}...")
                input_tensor_idx = op_detail['inputs'][0]
                weights_tensor_idx = op_detail['inputs'][1]
                biases_tensor_idx = op_detail['inputs'][2] if len(op_detail['inputs']) > 2 else -1
                output_tensor_idx = op_detail['outputs'][0]

                weights_data = interpreter.get_tensor(weights_tensor_idx)
                biases_data = interpreter.get_tensor(biases_tensor_idx) if biases_tensor_idx != -1 else None
                input_tensor_details = next(t for t in tensor_details if t['index'] == input_tensor_idx)
                output_tensor_details = next(t for t in tensor_details if t['index'] == output_tensor_idx)
                weights_tensor_details = next(t for t in tensor_details if t['index'] == weights_tensor_idx)
                
                # Correctly extract all quantization parameters
                input_scale = input_tensor_details['quantization_parameters'].get('scales', [1.0])[0]
                input_zero_point = input_tensor_details['quantization_parameters'].get('zero_points', [0])[0]
                output_scale = output_tensor_details['quantization_parameters'].get('scales', [1.0])[0]
                output_zero_point = output_tensor_details['quantization_parameters'].get('zero_points', [0])[0]
                weights_scale = weights_tensor_details['quantization_parameters'].get('scales', [1.0])[0]
                weights_zero_point = weights_tensor_details['quantization_parameters'].get('zero_points', [0])[0]

                # Flatten weights and biases to C-style arrays
                flat_weights = weights_data.flatten()
                weights_c_array_content = ', '.join([f"{w}" for w in flat_weights])
                biases_values = [f"{b}" for b in biases_data] if biases_data is not None else []

                hf.write(f"// --- Layer {layer_idx} Parameters ---\n")
                hf.write(f"extern const int8_t L{layer_idx}_weights[{len(flat_weights)}];\n")
                if biases_data is not None:
                    hf.write(f"extern const int32_t L{layer_idx}_biases[{len(biases_values)}];\n")
                hf.write(f"extern const float L{layer_idx}_input_scale;\n")
                hf.write(f"extern const int32_t L{layer_idx}_input_zero_point;\n")
                hf.write(f"extern const float L{layer_idx}_output_scale;\n")
                hf.write(f"extern const int32_t L{layer_idx}_output_zero_point;\n")
                hf.write(f"extern const float L{layer_idx}_weights_scale;\n")
                hf.write(f"extern const int32_t L{layer_idx}_weights_zero_point;\n\n")

                sf.write(f"// Layer {layer_idx} Parameters\n")
                sf.write(f"const int8_t L{layer_idx}_weights[{len(flat_weights)}] = {{\n    {weights_c_array_content}\n}};\n")
                if biases_data is not None:
                    sf.write(f"const int32_t L{layer_idx}_biases[{len(biases_values)}] = {{\n    {', '.join(biases_values)}\n}};\n")
                sf.write(f"const float L{layer_idx}_input_scale = {input_scale:.8f}f;\n")
                sf.write(f"const int32_t L{layer_idx}_input_zero_point = {input_zero_point};\n")
                sf.write(f"const float L{layer_idx}_output_scale = {output_scale:.8f}f;\n")
                sf.write(f"const int32_t L{layer_idx}_output_zero_point = {output_zero_point};\n")
                sf.write(f"const float L{layer_idx}_weights_scale = {weights_scale:.8f}f;\n")
                sf.write(f"const int32_t L{layer_idx}_weights_zero_point = {weights_zero_point};\n\n")
                layer_idx += 1

        print("Generating quantized sample inputs and labels...")
        (_, _), (x_test_raw, y_test_raw) = tf.keras.datasets.mnist.load_data()
        
        # Preprocess and quantize the test inputs        
        # NOTE: Your preprocessing from the notebook must be replicated here.
        def preprocess_image_gen(image_28x28):
            image_float = image_28x28.astype('float32') / 255.0
            rows = np.any(image_float > 0.1, axis=1)
            cols = np.any(image_float > 0.1, axis=0)
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            buffer = 2
            ymin = max(0, ymin - buffer)
            ymax = min(28, ymax + buffer)
            xmin = max(0, xmin - buffer)
            xmax = min(28, xmax + buffer)
            cropped_image = image_float[ymin:ymax, xmin:xmax]
            resized_image = tf.image.resize(np.expand_dims(cropped_image, axis=-1), (12, 12), method='bilinear').numpy().squeeze()
            final_image_binary = (resized_image > 0.4).astype('float32')
            return final_image_binary

        x_test_preprocessed = np.array([preprocess_image_gen(img) for img in x_test_raw])
        x_test_flattened = x_test_preprocessed.reshape(-1, 144)

        # Get input quantization parameters from the TFLite model
        model_input_details = interpreter.get_input_details()[0]
        model_input_scale = model_input_details['quantization_parameters'].get('scales', [1.0])[0]
        model_input_zero_point = model_input_details['quantization_parameters'].get('zero_points', [0])[0]

        num_samples_to_generate = 10
        hf.write("// --- Quantized sample input images and their labels ---\n")
        sf.write("// Quantized sample input images and their labels\n")

        for i in range(num_samples_to_generate):
            original_image_float = x_test_flattened[i].flatten()
            quantized_image_int8 = quantize_input(original_image_float, model_input_scale, model_input_zero_point)
            label = y_test_raw[i]
            image_c_array_content = ', '.join([f"{val}" for val in quantized_image_int8])
            hf.write(f"extern const int8_t input_data_{i}[{len(quantized_image_int8)}];\n")
            hf.write(f"extern const uint8_t label_{i};\n")
            sf.write(f"const int8_t input_data_{i}[{len(quantized_image_int8)}] = {{\n    {image_c_array_content}\n}};\n")
            sf.write(f"const uint8_t label_{i} = {label};\n")

        hf.write("\n#endif // MNIST_MODEL_PARAMS_H\n")
    print(f"Generated {header_file} and {source_file} with corrected model parameters and sample inputs.")

if __name__ == "__main__":
    if not os.path.exists(TFLITE_MODEL_PATH):
        print(f"Error: Quantized TFLite model not found at {TFLITE_MODEL_PATH}.")
    else:
        generate_c_arrays_from_tflite(TFLITE_MODEL_PATH, C_HEADER_FILE, C_SOURCE_FILE)