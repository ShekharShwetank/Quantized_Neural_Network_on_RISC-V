#ifndef MNIST_MODEL_PARAMS_H
#define MNIST_MODEL_PARAMS_H

#include <stdint.h>

#define MAX_N_ACTIVATIONS 144
#define ACTIVATION_BITS 8

// --- Layer 1 Parameters ---
extern const int8_t L1_weights[9216];
extern const int32_t L1_biases[64];
extern const float L1_input_scale;
extern const int32_t L1_input_zero_point;
extern const float L1_output_scale;
extern const int32_t L1_output_zero_point;
extern const float L1_weights_scale;
extern const int32_t L1_weights_zero_point;

// --- Layer 2 Parameters ---
extern const int8_t L2_weights[4096];
extern const int32_t L2_biases[64];
extern const float L2_input_scale;
extern const int32_t L2_input_zero_point;
extern const float L2_output_scale;
extern const int32_t L2_output_zero_point;
extern const float L2_weights_scale;
extern const int32_t L2_weights_zero_point;

// --- Layer 3 Parameters ---
extern const int8_t L3_weights[640];
extern const int32_t L3_biases[10];
extern const float L3_input_scale;
extern const int32_t L3_input_zero_point;
extern const float L3_output_scale;
extern const int32_t L3_output_zero_point;
extern const float L3_weights_scale;
extern const int32_t L3_weights_zero_point;

// --- Quantized sample input images and their labels ---
extern const int8_t input_data_0[144];
extern const uint8_t label_0;
extern const int8_t input_data_1[144];
extern const uint8_t label_1;
extern const int8_t input_data_2[144];
extern const uint8_t label_2;
extern const int8_t input_data_3[144];
extern const uint8_t label_3;
extern const int8_t input_data_4[144];
extern const uint8_t label_4;
extern const int8_t input_data_5[144];
extern const uint8_t label_5;
extern const int8_t input_data_6[144];
extern const uint8_t label_6;
extern const int8_t input_data_7[144];
extern const uint8_t label_7;
extern const int8_t input_data_8[144];
extern const uint8_t label_8;
extern const int8_t input_data_9[144];
extern const uint8_t label_9;

#endif // MNIST_MODEL_PARAMS_H
