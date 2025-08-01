BitNet-like Quantized MNIST Classification on VSDSquadron PRO
This repository contains the embedded C application and associated Python tools for deploying a quantized MNIST handwritten digit classification model onto the VSDSquadron PRO development board, which features the SiFive FE310-G002 RISC-V microcontroller. The solution uses a custom, efficient, integer-only inference engine inspired by BitNet principles.

## Features
* Model Architecture: A simple feed-forward neural network with three dense layers is used for MNIST classification. The layer sizes are 784 (input), 32 (hidden), 16 (hidden), and 10 (output) neurons.

* Quantized Inference: The model weights and activations are quantized to 8-bit integers, allowing for efficient, integer-only matrix multiplication on the RISC-V processor.

* RISC-V Compatibility: The C code is optimized for the RV32IMAC instruction set of the SiFive FE310-G002 SoC.

* Parameter Generation: A Python script (generate_c_model_params.py) extracts the quantized weights, biases, and sample inputs from a TensorFlow Lite (.tflite) model into C-compatible header and source files (mnist_model_params.h/.c).

* Bare-Metal Execution: The application runs directly on the microcontroller without an operating system, with inference results printed to a serial terminal for verification.

## Hardware Requirements
* VSDSquadron PRO Development Board: Featuring the SiFive FE310-G002 RISC-V SoC.

* USB-C Cable: For power, programming, and serial communication.

## Software Requirements
* Freedom Studio 3.1.1: The IDE for SiFive RISC-V development.

* Python 3.x: For the model parameter generation script.

* Python Libraries: tensorflow (version 2.15.0 or compatible) and numpy.

## Project Structure

```
.
├── src/
│   ├── app_inference.h              # Core C inference functions
│   ├── main.c                       # Main application logic and inference pipeline
│   ├── mnist_model_data.h           # TFLite model data as a C byte array
│   ├── mnist_model_params.c         # Generated C source file with model parameters
│   ├── mnist_model_params.h         # Generated C header with extern declarations
│   ├── mnist_quantized_model.tflite # Quantized TensorFlow Lite model binary
│   ├── generate_c_model_params.py   # Python script to generate model parameters
│   └── Makefile                     # Project-specific Makefile
├── mnist_baseline_model.ipynb       # Jupyter notebook for model training and quantization
└── ... (Other SDK-related files)
```

## Workflow
1. Generate Quantized Model: Run the `mnist_baseline_model.ipynb` Jupyter notebook to train the model and generate `mnist_quantized_model.tflite`.

2. Generate C Parameters: Execute python generate_c_model_params.py in the `src/` directory to create `mnist_model_params.c` and `mnist_model_params.h` from the `.tflite` file.

3. Build and Flash: In Freedom Studio, build the C application, connect the VSDSquadron PRO board via USB, and flash the compiled binary to the microcontroller.

4. Run Inference: The board will automatically begin inference, printing the results (predicted digit, true label, and status) to the serial terminal.

## OUTPUT:

```
BitNet MNIST Dataset Handwritten Digit Classification on sifive-hifive1.



By Shwetank Shekhar
Starting MNIST inference...
Processing input for sample 1
Debug: Starting first layer processing
Debug: First layer complete, applying ReLU
Debug: Starting second layer
Debug: Second layer complete, applying ReLU
Debug: Starting third layer
Debug: Third layer complete, scaling outputs
Predicted digit: 7, True Label: 7, Status: PASS

Processing input for sample 2
Debug: Starting first layer processing
Debug: First layer complete, applying ReLU
Debug: Starting second layer
Debug: Second layer complete, applying ReLU
Debug: Starting third layer
Debug: Third layer complete, scaling outputs
Predicted digit: 1, True Label: 2, Status: FAIL

Processing input for sample 3
Debug: Starting first layer processing
Debug: First layer complete, applying ReLU
Debug: Starting second layer
Debug: Second layer complete, applying ReLU
Debug: Starting third layer
Debug: Third layer complete, scaling outputs
Predicted digit: 1, True Label: 1, Status: PASS

Processing input for sample 4
Debug: Starting first layer processing
Debug: First layer complete, applying ReLU
Debug: Starting second layer
Debug: Second layer complete, applying ReLU
Debug: Starting third layer
Debug: Third layer complete, scaling outputs
Predicted digit: 0, True Label: 0, Status: PASS

Processing input for sample 5
Debug: Starting first layer processing
Debug: First layer complete, applying ReLU
Debug: Starting second layer
Debug: Second layer complete, applying ReLU
Debug: Starting third layer
Debug: Third layer complete, scaling outputs
Predicted digit: 4, True Label: 4, Status: PASS

Processing input for sample 6
Debug: Starting first layer processing
Debug: First layer complete, applying ReLU
Debug: Starting second layer
Debug: Second layer complete, applying ReLU
Debug: Starting third layer
Debug: Third layer complete, scaling outputs
Predicted digit: 1, True Label: 1, Status: PASS

Processing input for sample 7
Debug: Starting first layer processing
Debug: First layer complete, applying ReLU
Debug: Starting second layer
Debug: Second layer complete, applying ReLU
Debug: Starting third layer
Debug: Third layer complete, scaling outputs
Predicted digit: 4, True Label: 4, Status: PASS

Processing input for sample 8
Debug: Starting first layer processing
Debug: First layer complete, applying ReLU
Debug: Starting second layer
Debug: Second layer complete, applying ReLU
Debug: Starting third layer
Debug: Third layer complete, scaling outputs
Predicted digit: 1, True Label: 9, Status: FAIL

Processing input for sample 9
Debug: Starting first layer processing
Debug: First layer complete, applying ReLU
Debug: Starting second layer
Debug: Second layer complete, applying ReLU
Debug: Starting third layer
Debug: Third layer complete, scaling outputs
Predicted digit: 6, True Label: 5, Status: FAIL

Processing input for sample 10
Debug: Starting first layer processing
Debug: First layer complete, applying ReLU
Debug: Starting second layer
Debug: Second layer complete, applying ReLU
Debug: Starting third layer
Debug: Third layer complete, scaling outputs
Predicted digit: 4, True Label: 9, Status: FAIL
```

## POTENTIAL AREA OF IMPROVEMENT:

To improve the prediction accuracy of the model running on the VSDSquadron PRO board, several changes can be made to the Python training, quantization, and C inference code. The inaccuracies are likely due to the extreme quantization and potential data handling issues that cause a loss of information.

### 1. Model Training and Quantization (`mnist_baseline_model.ipynb`)

Modify the Jupyter Notebook to create a more robust model before quantization.

* **Increase Model Complexity**: The current model is very simple, with a hidden layer of only 32 neurons followed by a 16-neuron layer. A slightly larger model might capture more features of the data. Increasing the number of neurons in the dense layers could improve accuracy, provided the memory constraints of the `SiFive FE310-G002` are respected.

* **Hyperparameter Tuning**: Experiment with different optimizers, learning rates, and epochs to find a configuration that yields a higher validation accuracy. The current learning rate is reduced with plateaus, but a more aggressive schedule or a different optimizer like `RMSprop` could be more effective.

* **Advanced Quantization**: The quantization process in the notebook currently uses `tf.lite.Optimize.DEFAULT` with experimental 4-bit settings. This can be aggressive. You can modify the notebook to use standard 8-bit post-training quantization (`tf.lite.OpsSet.TFLITE_BUILTINS_INT8`) which is more reliable. The `generate_c_model_params.py` and `app_inference.h` files are already configured for 8-bit weights, so this change would align the entire toolchain.

### 2. C Inference Code (`app_inference.h`, `main.c`)

The C code can be optimized to improve accuracy by reducing potential data loss and miscalculations.

* **Input Scaling**: The provided `main.c` does not explicitly use the scale and zero-point from the `mnist_model_params.c` file. The input data is assumed to be correctly quantized, but re-quantizing it in C using the exact TFLite parameters can prevent mismatches. The formula `quantized_data = round(R / S + Z)` should be implemented to ensure inputs are handled correctly.

* **Intermediate Scaling**: The `ReLUNorm` function scales the output of each layer to fit into an 8-bit range. The scaling factor, determined by finding the max value, might not be optimal. A fixed scaling factor or a more sophisticated approach could be implemented to preserve more information.

### 3. Debugging and Profiling

* **Inspect Intermediate Values**: Adding statements to display the output of `layer1_out` and `layer2_out` to see if the values are meaningful or if they are getting saturated (e.g., all 0 or 127). This will pinpoint which layer is causing the most significant loss of information.

* **Compared to Python Output**: Run the Python model with the same input samples and save the intermediate output values. Compared these values with the C code's intermediate outputs to identify where the numerical divergence occurs.
