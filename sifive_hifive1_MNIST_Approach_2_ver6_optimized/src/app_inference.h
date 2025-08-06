/*
    BitNetMCU inference functions
    @cpldcpu April 2024

    Performs inference on fully connected layer on a very resource constrained MCU.
    
    C:\VSD_Sqd_Project\sifive_hifive1_BitNet_MNIST_App\src\app_inference.h
*/
#ifndef BITNETMCU_INFERENCE_H
#define BITNETMCU_INFERENCE_H


#include <stdint.h>
#include <stdio.h>
#include "mnist_model_params.h"

// Note: BATCH_SIZE and other parameters are now defined in mnist_model_params.h
// #define BATCH_SIZE 16

/**
 * @brief Applies a ReLU activation function to an array of integers and normalizes the result to 8-bit integers.
 *
 * @param input Pointer to the input array of 32-bit integers.
 * @param output Pointer to the output array of 8-bit integers.
 * @param n_input The number of elements in the input array.
 * @return The position of maximum value found in the input array before applying the ReLU activation.
 */

uint32_t ReLUNorm(int32_t *input, int8_t *output, uint32_t n_input) {
    int32_t max_val = 0; // Initialize to 0. For ReLU, we only care about positive max values.
    uint32_t max_pos = 0; 

    for (uint32_t i = 0; i < n_input; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
            max_pos = i;
        }
    }

    if (max_val <= 0) {
        for (uint32_t i = 0; i < n_input; i++) {
            output[i] = 0;
        }
        return max_pos;
    }

    uint32_t shift = 0;
    uint32_t temp_max_val = (uint32_t)max_val; 

    while (temp_max_val > 127) {
        temp_max_val >>= 1;
        shift++;
    }

    int32_t rounding = (shift > 0) ? (1 << (shift - 1)) : 0;

    for (uint32_t i = 0; i < n_input; i++) {
        if (input[i] < 0) {
            output[i] = 0;
        } else {
            int32_t scaled_val = (input[i] + rounding) >> shift;

            if (scaled_val > 127) {
                output[i] = 127;
            } else {
                output[i] = (int8_t)scaled_val;
            }
        }
    }
    return max_pos;
}

/**
 * @brief Processes a fully connected layer with 8-bit quantized weights.
 *
 * @param activations Pointer to the input activations.
 * @param weights Pointer to the packed 32-bit weights.
 * @param biases Pointer to the biases.
 * @param n_input The number of input neurons.
 * @param n_output The number of output neurons.
 * @param output Pointer to the output array (32-bit sum).
 */
void processfclayer(int8_t *activations, const uint32_t *weights,
                   const int32_t *biases, uint32_t n_input, uint32_t n_output, int32_t *output) {
    // This corrected implementation assumes 8-bit weights packed into uint32_t
    // where each uint32_t contains 4 int8 weights.
    // The original code was overly complex and had incorrect indexing for weights.

    for (uint32_t o = 0; o < n_output; o++) {
        int32_t sum = biases[o];

        // Process all inputs for the current output neuron
        for (uint32_t i = 0; i < n_input; i += 4) {
            uint32_t packed_weights = weights[(o * n_input / 4) + (i / 4)];
            int8_t weight_0 = (int8_t)((packed_weights >> 24) & 0xFF);
            int8_t weight_1 = (int8_t)((packed_weights >> 16) & 0xFF);
            int8_t weight_2 = (int8_t)((packed_weights >> 8) & 0xFF);
            int8_t weight_3 = (int8_t)(packed_weights & 0xFF);

            sum += (int32_t)activations[i] * weight_0;
            if ((i + 1) < n_input) sum += (int32_t)activations[i+1] * weight_1;
            if ((i + 2) < n_input) sum += (int32_t)activations[i+2] * weight_2;
            if ((i + 3) < n_input) sum += (int32_t)activations[i+3] * weight_3;
        }

        output[o] = sum;
    }
}
#endif // BITNETMCU_INFERENCE_H