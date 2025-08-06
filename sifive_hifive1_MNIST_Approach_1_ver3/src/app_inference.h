/*
    BitNetMCU inference functions
    @cpldcpu April 2024

    Performs inference on fully connected layer on a very resource constrained MCU.
    1,2,4 bit weights are supported.

    C:\VSD_Sqd_Project\sifive_hifive1_BitNet_MNIST_App\src\app_inference.h
*/
#ifndef BITNETMCU_INFERENCE_H
#define BITNETMCU_INFERENCE_H


#include <stdint.h>
#include <stdio.h>
#include "mnist_model_data.h"

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
                         // If all inputs are negative, max_val should remain 0 after this loop.
    uint32_t max_pos = 0; // Initialize to a default valid index (e.g., 0)

    // Find the maximum positive value in the input array.
    // This loop identifies the maximum value that could potentially be scaled.
    for (uint32_t i = 0; i < n_input; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
            max_pos = i; // Store the index of this maximum value
        }
    }

    // If the maximum value found is 0 or negative, it means all inputs to ReLU are non-positive.
    // In such a case, all outputs after ReLU should be 0.
    if (max_val <= 0) {
        for (uint32_t i = 0; i < n_input; i++) {
            output[i] = 0; // Set all output elements to 0
        }
        return max_pos; // Return the position of the max_val, even if it's 0 or negative
    }

    // Calculate the optimal right shift amount to scale 'max_val' into the target 8-bit range [0, 127].
    // This is done by repeatedly right-shifting 'max_val' until it fits.
    uint32_t shift = 0;
    // Create a temporary variable to avoid modifying original max_val prematurely.
    uint32_t temp_max_val = (uint32_t)max_val; // Cast to unsigned for logical shifts

    // Find the minimum 'shift' value such that 'temp_max_val' fits within 7 bits (0-127 range, leaving sign bit for int8_t)
    while (temp_max_val > 127) {
        temp_max_val >>= 1; // Equivalent to dividing by 2
        shift++;
    }

    // Calculate rounding value for "round to nearest" behavior in integer division.
    // If there's no shift (shift == 0), no rounding is needed as there's no division.
    // Otherwise, rounding is half of the smallest unit (1 << (shift - 1)).
    int32_t rounding = (shift > 0) ? (1 << (shift - 1)) : 0;

    // Apply ReLU activation and then scale the input values to the 8-bit output range.
    for (uint32_t i = 0; i < n_input; i++) {
        if (input[i] < 0) {
            output[i] = 0; // ReLU: Negative inputs become zero
        } else {
            // Apply scaling with the calculated shift and rounding.
            int32_t scaled_val = (input[i] + rounding) >> shift;

            // Clip the scaled value to fit within the valid signed 8-bit range [0, 127] (since inputs are non-negative after ReLU).
            if (scaled_val > 127) {
                output[i] = 127; // Clip to max positive 8-bit value
            } else {
                output[i] = (int8_t)scaled_val; // Cast to signed 8-bit integer
            }
        }
    }
    return max_pos; // Return the index of the detected maximum value.
}
/**
 * @brief Processes a fully connected layer in a neural network.
 *
 * This function processes a fully connected layer in a neural network by performing
 * the dot product of the input activations and weights, and stores the result in the output array.
 *
 * @param activations Pointer to the input activations of the layer.
 * @param weights Pointer to the weights of the layer.
 * @param bits_per_weight The number of bits per weight.
 * @param n_input The number of input neurons.
 * @param n_output The number of output neurons.
 * @param output Pointer to the output array where the result of the layer is stored.
 */

void processfclayer( int8_t *activations,  const uint32_t *weights, int32_t bits_per_weight, uint32_t n_input, uint32_t n_output, int32_t *output)
{
   const uint32_t *weightidx = weights;

    for (uint32_t i = 0; i < n_output; i++) {
        int8_t *activations_idx = activations;
        int32_t sum = 0;

        if (bits_per_weight == 1) {
            for (uint32_t k = 0; k < n_input; k+=32) {
                uint32_t weightChunk = *weightidx++;
                for (uint32_t j = 0; j < 32; j++) {
                    int32_t in=*activations_idx++;
                    sum += (weightChunk & 0x80000000) ? in : -in;  // Note that sign is flipped for Binary quant (bit set equals positive)
                    weightChunk <<= 1;
                }
            }
        } else if (bits_per_weight == 2 ) {
            for (uint32_t k = 0; k < n_input; k+=16) {
                uint32_t weightChunk = *weightidx++;
                for (uint32_t j = 0; j < 16; j++) {
                    int32_t in=*activations_idx++;
                    int32_t tmpsum = (weightChunk & 0x80000000) ? -in : in; // one complements sign (bit set equals negative)
                    sum += tmpsum;                                  // sign*in*1
                    if (weightChunk & 0x40000000) sum += tmpsum<<1; // sign*in*2
                    weightChunk <<= 2;
                }
            }
        // Multiplier-less inference for RV32EC
#if defined(__riscv) && !defined(__riscv_mul)
        } else if (bits_per_weight == 4 ) {
            for (uint32_t k = 0; k < n_input; k+=8) {
                uint32_t weightChunk = *weightidx++;
                for (uint32_t j = 0; j < 8; j++) {
                    int32_t in=*activations_idx++;
                    if (in != 0) { // Skip zero activations to speed up inference in layers after first layer
                        int32_t tmpsum = (weightChunk & 0x80000000) ? -in : in; // one complements sign (bit set equals negative)
                        sum += tmpsum;                                  // sign*in*1
                        if (weightChunk & 0x10000000) sum += tmpsum<<1; // sign*in*2
                        if (weightChunk & 0x20000000) sum += tmpsum<<2; // sign*in*4
                        if (weightChunk & 0x40000000) sum += tmpsum<<3; // sign*in*8
                    }
                    weightChunk <<= 4;
                }
            }
#else
        } else if (bits_per_weight == 4) { // 4 bit symmetric
            for (uint32_t k = 0; k < n_input; k+=8) { // Iterating 8 elements (4 bits each) per uint32_t chunk
                uint32_t weightChunk = *weightidx++; // Retrieve the 32-bit chunk containing 8 packed 4-bit weights
                for (uint32_t j = 0; j < 8; j++) { // Process each of the 8 individual 4-bit weights within the chunk
                    int32_t in = *activations_idx++; // Get the current input activation (int8_t, extended to int32_t)

                    // Extract the current 4-bit weight from the most significant bits of weightChunk.
                    // (weightChunk >> 28) brings the relevant 4-bit group to the least significant position.
                    // (& 0xF) masks to isolate only those 4 bits.
                    uint8_t nibble = (uint8_t)((weightChunk >> 28) & 0xF);

                    // Perform manual sign extension from 4 bits to 32 bits.
                    // For a 4-bit signed number, if the most significant bit (0x8) is set, it's a negative number.
                    int32_t weight;
                    if (nibble & 0x8) { // Check if the highest bit of the 4-bit value is set (indicating negative)
                        weight = (int32_t)(nibble | 0xFFFFFFF0); // Sign extend: fill the upper 28 bits with ones (0xFFFFFFF0)
                    } else {
                        weight = (int32_t)nibble; // Positive number, no sign extension needed
                    }

                    sum += in * weight; // Perform the multiply-accumulate operation
                    weightChunk <<= 4; // Shift the chunk left by 4 bits to bring the next 4-bit weight to the MSB position for the next iteration
                }
            }
        } else if (bits_per_weight == 8 + 4 ) {   // 4 bit twos-complement
            for (uint32_t k = 0; k < n_input; k+=8) {
                int32_t weightChunk = *weightidx++;
                for (uint32_t j = 0; j < 8; j++) {
                    int32_t in=*activations_idx++;
                    int32_t weight = (weightChunk) >> (32-4); // extend sign, cut off lower bits
                    sum += in*weight;
                    weightChunk <<= 4;
                }
            }
        } else if (bits_per_weight == 8 + 8 ) {   // 8 bit twos-complement
            for (uint32_t k = 0; k < n_input; k+=4) {
                int32_t weightChunk = *weightidx++;
                for (uint32_t j = 0; j < 4; j++) {
                    int32_t in=*activations_idx++;
                    int32_t weight = (weightChunk) >> (32-8); // extend sign, cut off lower bits
                    sum += in*weight;
                    weightChunk <<= 8;
                }
            }
#endif
        }  else if (bits_per_weight == 16 + 4 ) {  // 4 bit shift
            for (uint32_t k = 0; k < n_input; k+=8) {
                uint32_t weightChunk = *weightidx++;
                for (uint32_t j = 0; j < 8; j++) {
                    int32_t in=*activations_idx++;
                    int32_t tmpsum;

                    tmpsum = (weightChunk & 0x80000000) ? -in : in; // one complements sign (bit set equals negative)
                    sum += tmpsum << ((weightChunk >> 28) & 7); // sign*in*2^log
                    weightChunk <<= 4;
                }
            }
        }   // else printf("Error: unsupported weight bit width %d\n", bits_per_weight);

        output[i] = sum;
        // printf("%d,", output[i]);
    }
}
#endif // BITNETMCU_INFERENCE_H
