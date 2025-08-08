import serial
import cv2
import numpy as np
import tensorflow as tf

# --- Configuration ---
# !!! IMPORTANT: Replace with your actual serial port and quantization values
SERIAL_PORT = 'COM10'  # Example: 'COM3' on Windows or '/dev/ttyUSB0' on Linux
BAUD_RATE = 115200

# Quantization parameters from your mnist_model_params.c file
INPUT_SCALE = 0.00392157
INPUT_ZERO_POINT = -128

# --- Preprocessing Function (from your other scripts) ---
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
    return final_image_binary

# --- Main script ---
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert to grayscale and resize to 28x28
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, (28, 28), interpolation=cv2.INTER_AREA)

        # Preprocess and quantize the image for the board
        preprocessed_image = preprocess_image(resized_frame)
        quantized_image = np.round(preprocessed_image / INPUT_SCALE + INPUT_ZERO_POINT).astype(np.int8)

        # Flatten and send data over UART
        image_bytes = quantized_image.flatten().tobytes()

        ser.write(b'\xAA')  # Send start byte
        ser.write(image_bytes)

        print("Image sent. Waiting for prediction...")

        # Read and print response from board
        response = ser.readline().decode('utf-8').strip()
        if response:
            print(f"Board response: {response}")

        cv2.imshow('Webcam Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except serial.SerialException as e:
    print(f"Serial Port Error: {e}")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("Serial port closed.")
    if 'cap' in locals():
        cap.release()
        cv2.destroyAllWindows()