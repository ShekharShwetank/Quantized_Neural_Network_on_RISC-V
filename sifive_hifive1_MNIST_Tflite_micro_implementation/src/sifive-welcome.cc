#include <stdio.h>
#include <string.h>

// TFLM Includes from your library structure
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_allocator.h" // ADD THIS INCLUDE

// Model and Test Data Includes
#include "mnist_model_data.h"
#include "captured_image_data.h"

// Provide an implementation for the TFLM DebugLog function using printf.
// This function is called by the TFLM runtime for debug logging.


// --- TFLM Global Variables ---
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 4 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// --- Application Setup ---
void setup() {
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    model = tflite::GetModel(mnist_model_int8_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        error_reporter->Report("Model schema version mismatch!");
        return;
    }

    // Resolver capacity increased to register additional ops used by the
    // MNIST model (FULLY_CONNECTED, SOFTMAX, RESHAPE, LEAKY_RELU).
    // Use a slightly larger capacity to leave room for future ops.
    static tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    // The model contains a RESHAPE operator used by the graph; register it.
    resolver.AddReshape();
    // The provided model uses LeakyReLU activations; register the kernel.
    resolver.AddLeakyRelu();

    // Some TFLM versions expose a MicroInterpreter constructor that takes a
    // pre-created MicroAllocator; others take the arena directly. The repo
    // contains MicroAllocator and MicroInterpreter variants; create the
    // allocator explicitly and pass it to the interpreter to avoid the model
    // double-allocation problem.
    tflite::MicroAllocator* allocator =
        tflite::MicroAllocator::Create(tensor_arena, kTensorArenaSize);
    if (!allocator) {
        error_reporter->Report("MicroAllocator::Create() failed.");
        return;
    }

    // The allocator-based constructor signature in this repo is:
    // MicroInterpreter(const Model*, const MicroOpResolver&, MicroAllocator*,
    //                 MicroResourceVariables* = nullptr,
    //                 MicroProfilerInterface* = nullptr)
    // Pass nullptr for resource_variables and profiler (we still have
    // `error_reporter` for model/version checks and other reporting needs).
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, allocator, /*resource_variables=*/nullptr,
        /*profiler=*/nullptr);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        error_reporter->Report("AllocateTensors() failed.");
        return;
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    printf("\n TFLM setup complete.\n");
}

// --- Main Inference Function ---
void run_inference(const int8_t* image_data) {
    if (input == nullptr || output == nullptr) {
        error_reporter->Report("Input or output tensor is null\n");
        return;
    }

    // Print some diagnostics before invoking so we can see shapes/sizes.
    printf("[DEBUG] input->bytes=%d\n", (int)input->bytes);
    if (input->dims) {
        printf("[DEBUG] input dims size=%d: ", input->dims->size);
        for (int d = 0; d < input->dims->size; ++d) printf("%d ", input->dims->data[d]);
        printf("\n");
    }

    // Copy provided image into the input buffer (be careful with sizes)
    int to_copy = (int)input->bytes;
    memcpy(input->data.int8, image_data, to_copy);

    // Print the first few input values for verification
    {
        int preview = to_copy < 20 ? to_copy : 20;
        printf("[DEBUG] input (first %d bytes): ", preview);
        for (int i = 0; i < preview; ++i) printf("%d ", (int)input->data.int8[i]);
        printf("\n");
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        error_reporter->Report("Invoke failed.");
        return;
    }

    // Diagnostics for output tensor
    printf("----------------------------------------\n");
    printf("[DEBUG] output->bytes=%d\n", (int)output->bytes);
    if (output->dims) {
        printf("[DEBUG] output dims size=%d: ", output->dims->size);
        for (int d = 0; d < output->dims->size; ++d) printf("%d ", output->dims->data[d]);
        printf("\n");
    }

    // Quantization params (if applicable)
    printf("[DEBUG] output quant scale=%f zero_point=%d\n", (double)output->params.scale, (int)output->params.zero_point);

    // Print raw quantized outputs and dequantized floats
    int out_count = 10; // expected for MNIST
    printf("[DEBUG] output (quantized / dequantized):\n");
    for (int i = 0; i < out_count; ++i) {
        int q = (int)output->data.int8[i];
        float deq = (q - output->params.zero_point) * output->params.scale;
        printf("  [%d] %d / %f\n", i, q, (double)deq);
    }

    int8_t max_val = -128;
    int predicted_digit = -1;
    for (int i = 0; i < out_count; i++) {
        if (output->data.int8[i] > max_val) {
            max_val = output->data.int8[i];
            predicted_digit = i;
        }
    }

    printf("----------------------------------------\n");
    printf("Inference Result:\n");
    printf("Predicted Digit: %d\n", predicted_digit);
    printf("----------------------------------------\n\n");
}

extern "C" int main(void) {
    printf("Starting TFLM MNIST Example\n");

    setup();

    printf("\n Running inference on `captured_image` (True Label: 8)...\n");
    run_inference(captured_image);

    printf("\n Inference complete. Program finished.\n");

    return 0;
}
