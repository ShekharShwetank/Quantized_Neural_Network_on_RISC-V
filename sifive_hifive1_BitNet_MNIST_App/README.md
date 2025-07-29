>> Author: Shwetank Shekhar
   Date: 29-07-2025

# BitNet-like Quantized MNIST Classification on VSDSquadron PRO (SiFive FE310-G002)

This section of repository contains the embedded C application and associated Python tooling for deploying a quantized MNIST handwritten digit classification model onto the VSDSquadron PRO development board, powered by the SiFive FE310-G002 RISC-V microcontroller. The solution implements a custom, highly efficient, integer-only inference engine inspired by BitNet principles.

## Aim

The primary objective of this project is to demonstrate the end-to-end process of taking a pre-trained and quantized neural network model from a high-level framework (TensorFlow/Keras), converting it for embedded deployment, and running inference directly on a bare-metal RISC-V microcontroller. This includes handling data preparation, model parameter extraction, and optimizing the inference kernel for resource-constrained environments.

## Features

* **Quantized Inference:** Efficient execution of an 8-bit quantized MNIST classification model.
* **RISC-V Compatibility:** Optimized C code for the RV32IMAC instruction set of the SiFive FE310-G002.
* **BitNet-like Operations:** Custom `processfclayer` and `ReLUNorm` functions for efficient integer-only matrix multiplication and activation with support for packed bit-weights (specifically 4-bit and 8-bit handling demonstrated).
* **Automated Parameter Generation:** Python script to extract quantized weights, biases, and other layer parameters directly from a TensorFlow Lite (`.tflite`) model into C-compatible arrays.
* **Bare-Metal Execution:** Runs directly on the microcontroller without an operating system.
* **Serial Output:** Inference results are printed to a serial terminal for verification.

## Hardware Requirements

* **VSDSquadron PRO Development Board:** Featuring the SiFive FE310-G002 RISC-V SoC.
* **USB-C Cable:** For power, programming, debugging, and serial communication.

## Software Requirements

* **Freedom Studio 3.1.1:** The integrated development environment (IDE) for SiFive RISC-V development.
    * Download and installation instructions can be found in the [VSDSquadron PRO User Guide](https://github.com/ShekharShwetank/VSDSquadron_Pro_Edge_AI_Research_Internship/blob/master/datasheet.pdf).
* **Python 3.x:** For running the model parameter generation script.
* **Python Libraries:**
    * `tensorflow` (version 2.15.0 used in development).
    * `numpy`.

## Project Structure

```

.
├── src/
│   ├── app\_inference.h               \# Core C inference functions (processfclayer, ReLUNorm)
│   ├── main.c                         \# Main application logic, inference orchestration
│   ├── mnist\_model\_data.h           \# Contains the raw mnist\_quantized\_model.tflite binary data (for reference)
│   ├── mnist\_model\_params.c         \# Generated C source file with actual model weights, biases, and parameters
│   ├── mnist\_model\_params.h         \# Generated C header with extern declarations for model parameters
│   ├── mnist\_quantized\_model.tflite \# The quantized TensorFlow Lite model binary
│   └── Makefile                       \# Project-specific Makefile
├── mnist\_baseline\_model.ipynb       \# Jupyter notebook for model training, quantization, and .tflite export
├── bsp/                               \# Board Support Package (BSP) - provided by Freedom Studio/SiFive
├── freedom-metal/                     \# Freedom Metal library - bare-metal abstraction layer
├── scripts/                           \# Utility scripts for SDK (e.g., openocdcfg-generator)
└── ... (other SDK-related files and directories)

```

## Getting Started

Follow these steps to set up the development environment, generate model parameters, build the application, and run inference on your VSDSquadron PRO board.

### 1. Install Freedom Studio

Refer to the official [VSDSquadron PRO User Guide](https://github.com/ShekharShwetank/VSDSquadron_Pro_Edge_AI_Research_Internship/blob/master/datasheet.pdf) for detailed instructions on downloading, installing, and setting up Freedom Studio 3.1.1, including driver installation (e.g., using Zadig).

### 2. Prepare Python Environment

Ensure you have Python 3.x installed and the necessary libraries.

```bash
pip install tensorflow numpy
```

### 3\. Generate C Model Parameters and Sample Inputs

The `mnist_baseline_model.ipynb` notebook trains and quantizes the MNIST model, saving it as `mnist_quantized_model.tflite`. The `generate_c_model_params.py` script then extracts the layer-specific data from this `.tflite` file into C arrays.

1.  **Run Jupyter Notebook:** Open and run all cells in `mnist_baseline_model.ipynb` to ensure `mnist_quantized_model.tflite` is generated.

2.  **Run Parameter Generation Script:** Navigate to the `src/` directory in your terminal (e.g., `C:\VSD_Sqd_Project\sifive_hifive1_BitNet_MNIST_App\src\`) and execute:

    ```bash
    python generate_c_model_params.py
    ```

    This will create (or update) `mnist_model_params.h` and `mnist_model_params.c` in the `src/` directory. These files contain the quantized weights, biases, scales, zero points, and also the quantized sample input images and their labels.

### 4\. Build the Embedded Application

1.  **Open Project in Freedom Studio:** Launch Freedom Studio and import the `sifive_hifive1_BitNet_MNIST_App` project.

2.  **Clean Project:** In Freedom Studio, go to `Project -> Clean...`, select your project, and click `Clean`.

3.  **Build Project:** In Freedom Studio, go to `Project -> Build Project` or click the hammer icon in the toolbar.

      * **Expected Output:** The build should complete with `0 errors` and possibly `1 warning` (related to `RWX permissions` which is common in development). This indicates successful compilation and linking of all C source files, including the generated `mnist_model_params.c`.

### 5\. Configure and Run on VSDSquadron PRO

1.  **Connect Board:** Connect your VSDSquadron PRO board to your computer via a USB-C cable.

2.  **Configure Debug Launch:**

      * In Freedom Studio, go to `Run -> Debug Configurations...`.
      * In the left pane, expand `GDB OpenOCD Debugging` and select your project's launch configuration (e.g., `sifive_hifive1_BitNet_MNIST_App Debug`).
      * In the "Main" tab, ensure the "C/C++ Application" field points to the correct executable:
        ```
        ${workspace_loc:/sifive_hifive1_BitNet_MNIST_App/src/debug/main.elf}
        ```
        If it points to `empty.elf` or another path, correct it using the "Browse..." button.
      * Click `Apply` to save the changes.

3.  **Start Debug Session:** Click the `Debug` button in the Debug Configurations dialog, or the green bug icon in the toolbar.

      * If prompted to terminate a previous debug session, confirm "Yes".
      * The debugger will connect to the board and load the `main.elf` program. It will likely pause at the `main` function.

4.  **Run Program:** Click the `Resume` (green play) button to let the program execute.

5.  **Observe Output:** Monitor the "Console" or "Serial Terminal" view in Freedom Studio. You should see output similar to:

    ```
    BitNet MNIST Dataset Handwritten Digit Classification on sifive-hifive1.



    By Shwetank ShekharStarting MNIST inference...
    Inference of Sample 1   Prediction: 7   Label: 7
    Inference of Sample 2   Prediction: 2   Label: 2
    Inference of Sample 3   Prediction: 0   Label: 1
    Inference of Sample 4   Prediction: 0   Label: 0
    ```

    This confirms that the model is running on the hardware and performing inferences.

## Model Details

The model used is a simple feed-forward neural network for MNIST handwritten digit classification, trained using Keras and then quantized to 8-bit integers using TensorFlow Lite.

  * **Input Layer:** Flatten (28x28 grayscale image) -\> 784 features.
  * **Hidden Layer:** Dense layer with 8 neurons, ReLU activation.
  * **Output Layer:** Dense layer with 10 neurons (for 10 digits), Softmax activation.

The C inference engine (`app_inference.h`) is specifically tailored to handle 4-bit symmetric and 8-bit two's complement quantized weights, common in highly optimized embedded neural networks.

## Troubleshooting / Common Issues

For detailed explanations of specific errors encountered during development and their resolutions, please refer to the comprehensive [Error Resolution Log](sifive_hifive1_BitNet_MNIST_App/Error_Resolution.md).

Common issues addressed include:

  * Incorrect compiler attributes for functions.
  * Missing or undeclared model parameters in C code.
  * Python script errors due to TensorFlow Lite API changes (e.g., `AttributeError`, `KeyError`, `ModuleNotFoundError`).
  * Runtime hangs on the embedded target due to incorrect quantized arithmetic (e.g., 4-bit sign extension, `ReLUNorm` scaling issues).
  * Incorrect debug launch configurations in Freedom Studio.

## License

This project is licensed under the Apache 2.0 License - see the `LICENSE` file for details.

## Acknowledgments

  * VLSI System Design (VSD) & Mawle Technologies for the VSDSquadron PRO board and the internship opportunity.
  * SiFive for the FE310-G002 RISC-V SoC and Freedom Studio.
  * TensorFlow Lite team for the quantization and inference tools.
