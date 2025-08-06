/* Copyright 2019 SiFive, Inc */
/* SPDX-License-Identifier: Apache-2.0 */
#include <stdio.h>
#include <metal/cpu.h>
#include <metal/led.h>
#include <metal/button.h>
#include <metal/switch.h>

#include "app_inference.h"
#include "mnist_model_data.h"
#include "mnist_model_params.h"

#define RTC_FREQ 32768
#define MAX_N_ACTIVATIONS 64
#define DEBUG_PRINTS 1

// Helper function for software delay
static inline void software_delay(volatile int cycles) {
    for (volatile int i = 0; i < cycles; i++) {
        __asm__("nop");
    }
}

void display_banner (void) {
    printf("\n");
    printf("\n");
    printf("BitNet MNIST Dataset Handwritten Digit Classification on sifive-hifive1.\n");
    printf("\n");
    printf("\n");
    printf("\n");
    printf("By Shwetank Shekhar\n");
}

void BitMnistInference(const int8_t *input, const uint8_t label, const uint8_t sample) {
    // Corrected to use local arrays to avoid static VLA error
    // MAX_N_ACTIVATIONS is sufficient for layer 2 output (16) and layer 3 output (10)
    int32_t layer1_sum[L1_outgoing_weights]; // 32 neurons
    int8_t layer1_out[L1_outgoing_weights];
    
    int32_t layer2_sum[L2_outgoing_weights]; // 16 neurons
    int8_t layer2_out[L2_outgoing_weights];
    
    int32_t layer3_sum[L3_outgoing_weights]; // 10 neurons
    uint32_t pred_digit;

    printf("Processing input for sample %d\n", sample);

    // --- Layer 1 Processing ---
    printf("Debug: Starting first layer processing\n");
    processfclayer((int8_t*)input, L1_weights, L1_biases, L1_incoming_weights, L1_outgoing_weights, layer1_sum);
    
    printf("Debug: First layer complete, applying ReLU\n");
    ReLUNorm(layer1_sum, layer1_out, L1_outgoing_weights);

    // --- Layer 2 Processing ---
    printf("Debug: Starting second layer\n");
    processfclayer(layer1_out, L2_weights, L2_biases, L2_incoming_weights, L2_outgoing_weights, layer2_sum);
    
    printf("Debug: Second layer complete, applying ReLU\n");
    ReLUNorm(layer2_sum, layer2_out, L2_outgoing_weights);
    
    // --- Layer 3 Processing (Final Output Layer) ---
    printf("Debug: Starting third layer\n");
    processfclayer(layer2_out, L3_weights, L3_biases, L3_incoming_weights, L3_outgoing_weights, layer3_sum);

    printf("Debug: Third layer complete, scaling outputs\n");
    
    // Final prediction using a custom scaling function (not ReLUNorm)
    int32_t max_val = layer3_sum[0];
    pred_digit = 0;
    
    for (int i = 1; i < L3_outgoing_weights; i++) {
        if (layer3_sum[i] > max_val) {
            max_val = layer3_sum[i];
            pred_digit = i;
        }
    }
    
    // Print results for validation
    printf("Predicted digit: %d, True Label: %d, Status: %s\n",
           pred_digit, label, (pred_digit == label) ? "PASS" : "FAIL");
    printf("\n");
}

int main (void) {
    display_banner();
    printf("Starting MNIST inference...\n");

    BitMnistInference((const int8_t*)input_data_0, label_0, 1);
    BitMnistInference((const int8_t*)input_data_1, label_1, 2);
    BitMnistInference((const int8_t*)input_data_2, label_2, 3);
    BitMnistInference((const int8_t*)input_data_3, label_3, 4);
    BitMnistInference((const int8_t*)input_data_4, label_4, 5);
    BitMnistInference((const int8_t*)input_data_5, label_5, 6);
    BitMnistInference((const int8_t*)input_data_6, label_6, 7);
    BitMnistInference((const int8_t*)input_data_7, label_7, 8);
    BitMnistInference((const int8_t*)input_data_8, label_8, 9);
    BitMnistInference((const int8_t*)input_data_9, label_9, 10);

    return 0;
}
