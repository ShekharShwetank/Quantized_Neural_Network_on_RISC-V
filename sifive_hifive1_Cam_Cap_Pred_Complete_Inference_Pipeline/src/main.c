/* Copyright 2019 SiFive, Inc */
/* SPDX-License-Identifier: Apache-2.0 */
/* 
    AUTHOR: Shwetank Shekhar 
*/

#include <stdio.h>
#include <string.h>
#include <metal/cpu.h>
#include <metal/led.h>
#include <metal/button.h>
#include <metal/switch.h>

// COMMENT WHEN YOU DONT WANT TO USE UART
#include <metal/uart.h>

// FOR TIMER
#include <metal/timer.h>

// int metal_uart_read(struct metal_uart *uart, uint8_t *c, size_t length);

#include "app_inference.h"
#include "mnist_model_params.h"

// Include the header file for captured image data
#include "captured_image_data.h"

// 12x12=144.
#define L1_incoming_weights 144
#define L1_outgoing_weights 64
#define L2_incoming_weights 64
#define L2_outgoing_weights 64
#define L3_incoming_weights 64
#define L3_outgoing_weights 10

void software_delay(volatile int cycles) {
    for (volatile int i = 0; i < cycles * 100; i++) {
        __asm__("nop");
    }
}

void display_banner(void) {
    printf("\n");
    printf("\n");
    printf("Quantized TFLite MNIST on SiFive HiFive1.\n");
    printf("\n");
    printf("\n");
    printf("\n");
    printf("By Shwetank Shekhar\n");
}

void QInt8Inference(const int8_t *input, const uint8_t label, const uint8_t sample) {
    static int32_t layer1_sum[L1_outgoing_weights];
    static int8_t layer1_out[L1_outgoing_weights];
    static int32_t layer2_sum[L2_outgoing_weights];
    static int8_t layer2_out[L2_outgoing_weights];
    static int32_t layer3_sum[L3_outgoing_weights];
    uint32_t pred_digit;

    printf("Clearing arrays...\n");
    memset(layer1_sum, 0, sizeof(layer1_sum));
    memset(layer1_out, 0, sizeof(layer1_out));
    memset(layer2_sum, 0, sizeof(layer2_sum));
    memset(layer2_out, 0, sizeof(layer2_out));
    memset(layer3_sum, 0, sizeof(layer3_sum));

    printf("Processing input for sample %d\n", sample);
    software_delay(1000);

    unsigned long long start_time;
    unsigned long long end_time;
    unsigned long long timebase_freq;
    
    metal_timer_get_timebase_frequency(0, &timebase_freq);
    metal_timer_get_cyclecount(0, &start_time);

    // Layer 1 Processing
    printf("Starting first layer...\n");
    processfclayer(input, L1_weights, L1_biases, L1_incoming_weights, L1_outgoing_weights, layer1_sum,
                   L1_input_zero_point, L1_weights_zero_point);
    software_delay(1000);

    printf("Applying ReLU and Requantizing first layer...\n");
    quantized_relu_requantize(layer1_sum, layer1_out, L1_outgoing_weights, 
                             L1_output_scale, L2_input_scale, L2_input_zero_point);
    software_delay(1000);

    // Debug prints
    int min1 = 0, max1 = 0;
    for(int i=0; i<L1_outgoing_weights; i++) {
        if(layer1_out[i]<min1) min1=layer1_out[i];
        if(layer1_out[i]>max1) max1=layer1_out[i];
    }
    printf("Layer1 ReLU range: %d to %d\n", min1, max1);
    printf("Layer1 sample activations: %d %d %d %d\n", layer1_out[0], layer1_out[1], layer1_out[2], layer1_out[3]);

    // Layer 2 Processing
    printf("Starting second layer...\n");
    processfclayer(layer1_out, L2_weights, L2_biases, L2_incoming_weights, L2_outgoing_weights, layer2_sum,
                   L2_input_zero_point, L2_weights_zero_point);
    software_delay(1000);

    printf("Applying ReLU and Requantizing second layer...\n");
    quantized_relu_requantize(layer2_sum, layer2_out, L2_outgoing_weights, 
                             L2_output_scale, L3_input_scale, L3_input_zero_point);
    software_delay(1000);

    // Debug prints
    int min2 = 0, max2 = 0;
    for(int i=0; i<L2_outgoing_weights; i++) {
        if(layer2_out[i]<min2) min2=layer2_out[i];
        if(layer2_out[i]>max2) max2=layer2_out[i];
    }
    printf("Layer2 ReLU range: %d to %d\n", min2, max2);
    printf("Layer2 sample activations: %d %d %d %d\n", layer2_out[0], layer2_out[1], layer2_out[2], layer2_out[3]);

    // Layer 3 Processing (Output Layer)
    printf("Starting final layer...\n");
    processfclayer(layer2_out, L3_weights, L3_biases, L3_incoming_weights, L3_outgoing_weights, layer3_sum,
                   L3_input_zero_point, L3_weights_zero_point);
    software_delay(1000);

    // Before prediction, print all output values
    printf("Output layer values: ");
    for(int i=0; i<L3_outgoing_weights; i++) {
        printf("%ld ", layer3_sum[i]);
    }
    printf("\n");

    metal_timer_get_cyclecount(0, &end_time);
    unsigned long long inference_time_us = (end_time - start_time) * 1000000ULL / timebase_freq;
    printf("Inference completed in %llu us.\n", inference_time_us);

    // Find prediction
    printf("Finding prediction...\n");
    int32_t max_val = layer3_sum[0];
    pred_digit = 0;
    
    for (int i = 1; i < L3_outgoing_weights; i++) {
        if (layer3_sum[i] > max_val) {
            max_val = layer3_sum[i];
            pred_digit = i;
        }
        software_delay(100);
    }
    
    printf("Predicted digit: %ld, True Label: %d, Status: %s\n",
           pred_digit, label, (pred_digit == label) ? "PASS" : "FAIL");
    software_delay(1000);

    printf("\n\n");
}

// New input data for testing
const int8_t input_data_new[144] = {
    -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, 127, 127, 127, 127, 127, 127, 127, 127, 127, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, 127, 127, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, 127, 127, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, 127, 127, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, 127, 127, 127, 127, 127, 127, 127, 127, 127, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, 127, 127, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, 127, 127, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, 127, 127, 127, 127, 127, 127, 127, 127, 127, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128
};

// Function to read image data from UART
// UART is configured to receive 144 bytes of image data

/*
void read_image_from_uart(int8_t* buffer) {
    struct metal_uart *uart = metal_uart_get_device(0); // Get UART device handle
    uint8_t start_byte;

    // Wait for the start byte
    printf("Waiting for image data via UART...\n");
    while (metal_uart_read(uart, &start_byte, 1) != 1 || start_byte != 0xAA) {
        // Keep waiting
    }

    // Read 144 bytes of image data
    for (int i = 0; i < 144; i++) {
        while (metal_uart_read(uart, (uint8_t*)&buffer[i], 1) != 1) {
            // Keep waiting for byte
        }
    }
    printf("Image data received.\n");
}
*/

// UART WITH TIMER
void read_image_from_uart_with_timer(int8_t* buffer) {
    struct metal_uart *uart = metal_uart_get_device(0);
    int start_byte;
    unsigned long long start_time, end_time, timebase_freq;
    int read_char;

    metal_timer_get_timebase_frequency(0, &timebase_freq);
    metal_timer_get_cyclecount(0, &start_time);

    printf("Waiting for image data via UART...\n");

    // Wait for the start byte (0xAA) using metal_uart_getc
    do {
        metal_uart_getc(uart, &read_char);
    } while (read_char != 0xAA);
    start_byte = read_char; // Store the start byte

    // Read 144 bytes of image data
    for (int i = 0; i < 144; i++) {
        do {
            metal_uart_getc(uart, &read_char);
        } while (read_char == -1); // -1 indicates no character ready
        buffer[i] = (int8_t)read_char;
    }
    
    metal_timer_get_cyclecount(0, &end_time);
    unsigned long long reception_time_us = (end_time - start_time) * 1000000ULL / timebase_freq;
    printf("Image data received in %llu us.\n", reception_time_us);
}

int main(void) {
    display_banner();
    printf("Starting MNIST inference...\n");
    /*
     * Uncomment when you want to test the below:
     *
    QInt8Inference(input_data_0, label_0, 1);
    QInt8Inference(input_data_1, label_1, 2);
    QInt8Inference(input_data_2, label_2, 3);
    QInt8Inference(input_data_3, label_3, 4);
    QInt8Inference(input_data_4, label_4, 5);
    QInt8Inference(input_data_5, label_5, 6);
    QInt8Inference(input_data_6, label_6, 7);
    QInt8Inference(input_data_7, label_7, 8);
    QInt8Inference(input_data_8, label_8, 9);
    QInt8Inference(input_data_9, label_9, 10);
    */

    //TEST GENERATED IMAGE FROM THE generate_test_image.py
    // printf("Custom Test Image...\n");
    // QInt8Inference(input_data_new, 3, 11);

    // Uncomment to test with the captured image data
    // Test with the captured image
    printf("Testing with captured webcam image...\n");
    // The second parameter is the true label; we don't know it, so we'll use a placeholder (0)
    QInt8Inference(captured_image, 8, 12);

    // Uncomment for Real-time inference mode using UART
    /*
    printf("Starting MNIST inference in real-time mode...\n");

    int8_t input_buffer[144];
    while(1) {
        read_image_from_uart_with_timer(input_buffer);
        // Assuming we send a label along with the image for testing
        // For now, let's assume the label is known on the host side
        QInt8Inference(input_buffer, 0, 0); // Label and sample ID can be placeholder for now
    }
	*/
    return 0;
}

