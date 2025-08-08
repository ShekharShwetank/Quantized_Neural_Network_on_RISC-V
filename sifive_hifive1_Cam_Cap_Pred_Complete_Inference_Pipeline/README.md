# Capstone project: 

Capture images using your webcam, preprocess the image, send it over uart to board, then run onboard preprocessing, and then generate inference

### **Overview:**

1.  **Host-side (Python)**: A Python script captures live video from a webcam, preprocesses each frame, and sends the processed image data over a serial connection.
2.  **Embedded-side (C)**: A C program running on the SiFive HiFive1 continuously listens for incoming data, performs 8-bit quantized inference, and reports the predicted digit and performance metrics back to the host.

The entire pipeline, from image capture to prediction, is designed for low latency and minimal memory footprint, making it suitable for IoT and edge computing applications.

### **Directory Structure:**

```
.
├── src/
│   ├── app_inference.h                  # C header for quantized inference functions.
│   ├── captured_image_data.h            # Generated C header for static image testing.
│   ├── main.c                           # Main C program for on-board inference.
│   ├── mnist_model_data.h               # Generated C header for the TFLite model data.
│   ├── mnist_model_params.c             # Generated C source with quantized weights and biases.
│   ├── mnist_model_params.h             # Generated C header with model parameter declarations.
│   ├── cam_capture_image.py             # Python script for single-shot image capture and C array generation.
│   ├── captured_frame.png               # Saved image from webcam capture for testing.
│   ├── generate_c_model_params.py       # Python script to convert TFLite model to C arrays.
│   ├── generate_test_image.py           # Python script to generate synthetic C test arrays.
│   ├── Image_Processing.ipynb           # Jupyter Notebook for model training and TFLite conversion.
│   ├── LICENSE                          # Project license files.
│   ├── Makefile                         # GNU Makefile for building the embedded C code.
│   ├── mnist_baseline_model.ipynb       # Backup of the model training notebook.
│   ├── mnist_model_int8.tflite          # Final 8-bit quantized TensorFlow Lite model.
│   ├── processed_frame.png              # Saved preprocessed image.
│   ├── README.md                        # This file.
│   └── send_image_uart.py               # Python script for real-time UART image transmission.
└── ...
```

### **Prerequisites:**

  * **SiFive HiFive1 Board**: The physical hardware for running the embedded code.
  * **RISC-V GNU Toolchain**: The cross-compiler required for building C/C++ applications for the HiFive1.
  * **OpenOCD**: A debugging and flashing tool to upload the compiled binary to the board.
  * **Freedom Studio 3-1-1**: The IDE used for development.
  * **Python 3.x**: With `tensorflow`, `numpy`, `opencv-python`, and `pyserial` libraries installed.

### **Workflow:**

     1. Capture: Capture a live image of a handwritten digit using a laptop's webcam.

     2. Preprocessing: Apply all image preprocessing steps on the host to transform the raw image into a format suitable for the model.

     3. Transmission: Send the preprocessed image data over a serial connection (UART) to the SiFive HiFive1 board.
     
     4. Inference: The board receives the image data and performs the 8-bit quantized inference.
     
     5. Prediction: The board transmits the classification result back to the host computer via UART.
     
## **To Run Inference:**

     1. mnist_baseline_model.ipynb > generate_c_model_params.py > main.c
     2. cam_capture_image.py > main.c

### **Project Tasks and Optimizations:**

  * **Image Preprocessing**: Raw 28x28 images are processed on the host machine to crop the digit's bounding box and resize it to 12x12. This significantly reduces the input data size from 784 to 144 bytes, which improves inference speed and reduces communication latency.
  * **8-bit Quantization**: The trained model is converted from a floating-point format to an 8-bit integer format. This optimization drastically reduces the model's memory footprint, allowing it to fit into the HiFive1's constrained RAM, while also enabling faster integer arithmetic on the microcontroller.
  * **UART Communication**: A custom, low-level serial communication protocol is implemented to reliably send image data from the host to the board. A start byte is used to synchronize the data stream, ensuring that the board correctly processes each image.
  * **Performance Analysis**: The on-board C code includes a high-resolution timer to measure and report the time taken for both UART reception and the entire neural network inference process. This provides empirical data on the system's real-time performance.
  * **Robust Preprocessing**: The preprocessing functions include a fallback for cases where a digit is not detected, ensuring that the pipeline does not crash and can handle various image inputs reliably.

## **Inferences:**

Input:
     ![ip](src/captured_frame.png)

Output:

```
Quantized TFLite MNIST on SiFive HiFive1.



By Shwetank Shekhar
Starting MNIST inference...
Testing with captured webcam image...
Clearing arrays...
Processing input for sample 12
Starting first layer...
Processing layer: in=144, out=64
Applying ReLU and Requantizing first layer...
Layer1 ReLU range: -96 to 127
Layer1 sample activations: -96 127 127 127
Starting second layer...
Processing layer: in=64, out=64
Applying ReLU and Requantizing second layer...
Layer2 ReLU range: -103 to 127
Layer2 sample activations: -103 127 127 -103
Starting final layer...
Processing layer: in=64, out=10
Output layer values: -67787 -26102 -80181 -42641 -56793 -46513 -72808 -47695 -22804 -85658 
Inference completed in lu us.
Finding prediction...
Predicted digit: 8, True Label: 8, Status: PASS
```
