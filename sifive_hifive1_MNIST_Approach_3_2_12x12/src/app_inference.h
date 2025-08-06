/*
    Quantized TFLite inference functions
    Author: Shwetank Shekhar
    
    Performs inference on fully connected layers with standard 8-bit quantization.
*/

#ifndef QINT8_INFERENCE_H
#define QINT8_INFERENCE_H

#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include "mnist_model_params.h"

// Note: L1_incoming_weights must be 144 (12*12)
#define L1_incoming_weights 144
#define L1_outgoing_weights 64
#define L2_incoming_weights 64
#define L2_outgoing_weights 64
#define L3_incoming_weights 64
#define L3_outgoing_weights 10

/**
 * @brief Applies a ReLU activation and requantizes the output.
 *
 * @param input_sum Pointer to the input array of 32-bit sums.
 * @param output_q Pointer to the output array of 8-bit integers.
 * @param n_output The number of elements in the output array.
 * @param input_scale_float The scale of the input tensor.
 * @param input_zero_point_int32 The zero point of the input tensor.
 * @param output_scale_float The scale of the output tensor.
 * @param output_zero_point_int32 The zero point of the output tensor.
 */
void quantized_relu_requantize(const int32_t* input_sum, int8_t* output_q, uint32_t n_output,
                                float input_scale_float, float output_scale_float, int32_t output_zero_point_int32) {
    float requantization_scale = input_scale_float / output_scale_float;
    
    for (uint32_t i = 0; i < n_output; i++) {
        // Apply ReLU: clip negative values to 0
        int32_t relu_val = input_sum[i] > 0 ? input_sum[i] : 0;
        
        // Requantize the value
        float scaled_val = (float)relu_val * requantization_scale;
        int32_t quantized_val = (int32_t)round(scaled_val) + output_zero_point_int32;
        
        // Clamp the final value to the 8-bit range
        if (quantized_val > 127) {
            output_q[i] = 127;
        } else if (quantized_val < -128) {
            output_q[i] = -128;
        } else {
            output_q[i] = (int8_t)quantized_val;
        }
    }
}

/**
 * @brief Processes a fully connected layer with 8-bit quantized weights and activations.
 *
 * @param activations Pointer to the input activations (int8_t).
 * @param weights Pointer to the weights (int8_t).
 * @param biases Pointer to the biases (int32_t).
 * @param n_input The number of input neurons.
 * @param n_output The number of output neurons.
 * @param output Pointer to the output array (32-bit sum).
 * @param input_zero_point The zero point of the input tensor.
 * @param weights_zero_point The zero point of the weights tensor.
 */
void processfclayer(const int8_t* activations, const int8_t* weights, const int32_t* biases,
                    uint32_t n_input, uint32_t n_output, int32_t* output,
                    int32_t input_zero_point, int32_t weights_zero_point) {
    printf("Processing layer: in=%d, out=%d\n", n_input, n_output);

    for (uint32_t o = 0; o < n_output; o++) {
        int32_t sum = biases[o];
        for (uint32_t i = 0; i < n_input; i++) {
            int32_t a = (int32_t)activations[i] - input_zero_point;
            int32_t w = (int32_t)weights[o * n_input + i] - weights_zero_point;
            sum += a * w;
        }
        output[o] = sum;
    }
}

#endif // QINT8_INFERENCE_H
