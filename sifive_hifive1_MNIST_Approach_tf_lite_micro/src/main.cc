#include <stdio.h>
#include <string.h> // For memset

// TensorFlow Lite Micro includes
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h> // For MicroPrintf
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>

// If TFLITE_SCHEMA_VERSION is not defined by TFLM includes, define it here as a fallback.
// This is typically defined internally by TFLM for the target.
#ifndef TFLITE_SCHEMA_VERSION
#define TFLITE_SCHEMA_VERSION 3 // Common TFLite Micro schema version for compatibility
#endif

// Model data
#include "mnist_model_data.h"

// Standard C library includes for target (provided by Freedom Metal)
#include <metal/led.h> // Assuming freedom-metal provides LED control [cite: 3]
#include <metal/uart.h> // Assuming freedom-metal provides UART for output [cite: 4]

// Constants for the model
const int kTensorArenaSize = 1024; // 1KB - This is still VERY tight. Be prepared for OOM.

static uint8_t tensor_arena[kTensorArenaSize];

// Global variables for the model
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Metal UART and LED handles
struct metal_uart *uart0;
struct metal_led *blue_led;

void setup() {
    tflite::InitializeTarget();

    // Setup error reporter for logging
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    // Initialize UART for console output
    uart0 = metal_uart_get_device(0);// Assuming UART0 is available and configured [cite: 4]
    if (uart0 == NULL) {
        MicroPrintf("UART0 device not found!\n");
        // Handle error
    } else {
        metal_uart_init(uart0, 115200);// Initialize UART0 at 115200 baud [cite: 4]
        MicroPrintf("UART0 initialized for debug output.\n");
    }

    // Initialize Blue LED
    // The VSDSquadron PRO board uses GPIOs 19, 21, 22 for user LEDs. [cite: 2]
    // According to metal.h, LD0blue is mapped to pin 21. [cite: 1]
    blue_led = metal_led_get(const_cast<char*>("LD0blue"));// Use the correct API with label for blue LED [cite: 3]
    if (blue_led == NULL) {
        MicroPrintf("Blue LED device not found!\n");
    } else {
        MicroPrintf("Blue LED initialized.\n");
        metal_led_off(blue_led);// Start with LED off [cite: 3]
    }

    // Map the model into a usable data structure. This uses the schema_generated.h file.
    model = tflite::GetModel(mnist_quantized_model_tflite_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf("Model provided is schema version %d, not expected %d\n",
                   model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    // Create an OpResolver that will be used by the interpreter to find the
    // ops needed by the model. It is very important to only include the ops
    // that are actually used by your model to save valuable precious memory.
    static tflite::MicroMutableOpResolver<5> resolver; // Adjust template arg for number of ops
    resolver.AddFullyConnected(); // For Dense layer
    resolver.AddSoftmax();
    resolver.AddQuantize();     // For quantized input if needed
    resolver.AddDequantize();   // For dequantized output if needed
    resolver.AddReshape();      // For Flatten (Reshape is often part of Flatten's implementation)

    // Build an interpreter to run the model with.
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    // Allocate tensors from the arena.
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        MicroPrintf("Tensor allocation failed!\n");
        MicroPrintf("Needed: %d bytes\n", interpreter->arena_used_bytes());
        // This is where you'll likely see a failure if kTensorArenaSize is too small.
        return;
    }
    MicroPrintf("Tensor arena allocated: %d bytes used / %d bytes total.\n", interpreter->arena_used_bytes(), kTensorArenaSize);

    // Get information about the model's input and output tensors.
    input = interpreter->input(0);
    output = interpreter->output(0);

    // Ensure input tensor is INT8 as per your model quantization
    if (input->type != kTfLiteInt8) {
        MicroPrintf("Bad input tensor type %s, expected INT8\n", TfLiteTypeGetName(input->type));
        return;
    }

    MicroPrintf("Model setup complete. Ready for inference.\n");
}

void loop() {
    // This loop will continuously run inference or perform other tasks.
    // For a simple MNIST example, you might want to run inference on a predefined image.

    // Example: Prepare dummy input (a single 28x28 grayscale image as int8)
    // In a real application, this would come from a sensor or external source.
    int8_t dummy_image_data[28 * 28 * 1];
    memset(dummy_image_data, 0, sizeof(dummy_image_data)); // All pixels 0
    // You can fill this with a simple test pattern, e.g., a diagonal line or a simple digit
    // For example, to make a '1':
    // for(int i=0; i<28; ++i) dummy_image_data[i*28 + 14] = 127; // Simple vertical line

    // Copy the dummy image data to the input tensor
    if (input->bytes != (28 * 28 * 1)) {
         MicroPrintf("Input tensor size mismatch! Expected %d, got %d\n", 28*28*1, input->bytes);
         return;
    }
    memcpy(input->data.int8, dummy_image_data, input->bytes);


    // Run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        MicroPrintf("Invoke failed!\n");
        return;
    }

    // Process the output
    // Output tensor is int8, and its values are scaled.
    // To get actual probabilities, you need to dequantize if not handled by a Dequantize op.
    // For classification, often just finding the argmax (highest value) is enough.
    int8_t max_value = -128; // Smallest possible int8 value
    int predicted_class = -1;
    for (int i = 0; i < 10; ++i) { // Assuming 10 output classes
        if (output->data.int8[i] > max_value) {
            max_value = output->data.int8[i];
            predicted_class = i;
        }
    }

    MicroPrintf("Predicted digit: %d\n", predicted_class);
    metal_led_toggle(blue_led);// Toggle LED to show activity [cite: 3]
    for (volatile int i = 0; i < 500000; i++); // Simple delay

    // In a real application, you might only run inference on demand (e.g., button press).
    // For continuous execution in `loop`, just let it repeat.
}

// Standard entry point for freedom-metal applications
int main() {
    setup(); // Initialize model and peripherals

    while (true) {
        loop(); // Main application loop
    }

    return 0;
}
