/* Copyright 2019 SiFive, Inc */
/* SPDX-License-Identifier: Apache-2.0 */
// C:\VSD_Sqd_Project\sifive_hifive1_BitNet_MNIST_App\src\main.c
#include <stdio.h>
#include <metal/cpu.h>
#include <metal/led.h>
#include <metal/button.h>
#include <metal/switch.h>
void processfclayer(int8_t *, const uint32_t *, int32_t, uint32_t, uint32_t, int32_t *);

#include "app_inference.h"
#include "mnist_model_data.h"
#include "mnist_model_params.h"

#define RTC_FREQ    32768

struct metal_cpu *cpu;
struct metal_interrupt *cpu_intr, *tmr_intr;
int tmr_id;
volatile uint32_t timer_isr_flag;

#define MAX_N_ACTIVATIONS 64 // Max activations from Flatten layer (28*28)

/* void BitMnistInference(const int8_t *input, const uint8_t label, const uint8_t sample) {
    int32_t layer_out[MAX_N_ACTIVATIONS];
    int8_t layer_in[MAX_N_ACTIVATIONS];
	int32_t prediction;
	uint32_t startticks, endticks;

//	startticks = SysTick->CNT;
	// Placeholder definitions for L1, L2, L3 weights and parameters
	// In a real scenario, these would be generated from your .tflite model
	// and would be much larger arrays.

	// L1: Input 784, Output 8
	const uint32_t L1_weights[784 * 8 / 4] = {0}; // Placeholder
	const int32_t L1_bitperweight = 8 + 8;
	const uint32_t L1_incoming_weights = 784;
	const uint32_t L1_outgoing_weights = 8;

	// L2: Input 8, Output 10
	const uint32_t L2_weights[8 * 10 / 4] = {0}; // Placeholder
	const int32_t L2_bitperweight = 8 + 8;
	const uint32_t L2_incoming_weights = 8;
	const uint32_t L2_outgoing_weights = 10;

	// If you had L3 placeholders, remove them as well:
	// const uint32_t L3_weights[256 * 10 / 32] = {0}; // Placeholder
	// const int32_t L3_bitperweight = 1;
	// const uint32_t L3_incoming_weights = 256;
	// const uint32_t L3_outgoing_weights = 10;

	// And any NUM_LAYERS == 4 related placeholders
	// #ifdef NUM_LAYERS
	// #if NUM_LAYERS == 4
	// const uint32_t L4_weights[10 * 10 / 32] = {0}; // Placeholder
	// const int32_t L4_bitperweight = 1;
	// const uint32_t L4_incoming_weights = 10;
	// const uint32_t L4_outgoing_weights = 10;
	// #endif
	// #endif

//	endticks = SysTick->CNT;

	printf( "Inference of Sample %d\tPrediction: %ld\tLabel: %d\t\n", sample, prediction, label);
}
*/

void BitMnistInference(const int8_t *input, const uint8_t label, const uint8_t sample) {
    int32_t layer_out[MAX_N_ACTIVATIONS]; // Buffer for layer outputs (int32_t before ReLU)
    int8_t layer_in[MAX_N_ACTIVATIONS];    // Buffer for layer inputs (int8_t after ReLU/normalization)
    int32_t prediction;
    uint32_t startticks, endticks; // Timer variables (currently commented out)

    // Layer 1: Corresponds to Dense(8) in Python model
    // Input: `input` (quantized 784-element flattened image)
    // Output: `layer_out` (8 elements, before ReLU)
    processfclayer((int8_t*)input, L1_weights, L1_bitperweight, L1_incoming_weights, L1_outgoing_weights, layer_out);
    // Apply ReLU activation and normalize to 8-bit for next layer's input.
    // Output: `layer_in` (8 elements, after ReLU)
    ReLUNorm(layer_out, layer_in, L1_outgoing_weights);

    // Layer 2: Corresponds to Dense(10) in Python model
    // Input: `layer_in` (8 elements, output from previous ReLU)
    // Output: `layer_out` (10 elements, before final activation)
    processfclayer(layer_in, L2_weights, L2_bitperweight, L2_incoming_weights,  L2_outgoing_weights, layer_out);
    // Apply final activation (Softmax in Python, but ReLUNorm is used here to find max value for prediction)
    // Output: `layer_in` (10 elements, after final normalization, used only for finding max_pos for prediction)
    prediction = ReLUNorm(layer_out, layer_in, L2_outgoing_weights); // `ReLUNorm` returns max_pos, which is the predicted class index.

    // The original code had a conditional NUM_LAYERS == 4.
    // Since our Python model has only two Dense layers, this block is not needed.
    // #if NUM_LAYERS == 4
    //     processfclayer(layer_in, L4_weights, L4_bitperweight, L4_incoming_weights,  L4_outgoing_weights, layer_out);
    //     prediction=ReLUNorm(layer_out, layer_in, L4_outgoing_weights);
    // #endif

    // Print inference result
    printf( "Inference of Sample %d\tPrediction: %ld\tLabel: %d\t\n", sample, prediction, label);
}

void display_banner (void) {

    printf("\n");
    printf("\n");
    printf("BitNet MNIST Dataset Handwritten Digit Classification on sifive-hifive1.\n");
    printf("\n");
    printf("\n");

    printf("\n");
    printf("By Shwetank Shekhar");

}

void timer_isr (int id, void *data) {

    // Disable Timer interrupt
    metal_interrupt_disable(tmr_intr, tmr_id);

    // Flag showing we hit timer isr
    timer_isr_flag = 1;
}

void wait_for_timer(struct metal_led *which_led) {

    // clear global timer isr flag
    timer_isr_flag = 0;

    // Turn on desired LED
    metal_led_on(which_led);

    // Set timer
    metal_cpu_set_mtimecmp(cpu, metal_cpu_get_mtime(cpu) + RTC_FREQ);

    // Enable Timer interrupt
    metal_interrupt_enable(tmr_intr, tmr_id);

    // wait till timer triggers and isr is hit
    while (timer_isr_flag == 0){};

    timer_isr_flag = 0;

    // Turn off this LED
    metal_led_off(which_led);
}

int main (void)
{
    int rc;
    struct metal_led *led0_red, *led0_green, *led0_blue;

    // This demo will toggle LEDs colors so we define them here
    led0_red = metal_led_get_rgb("LD0", "red");
    led0_green = metal_led_get_rgb("LD0", "green");
    led0_blue = metal_led_get_rgb("LD0", "blue");
    if ((led0_red == NULL) || (led0_green == NULL) || (led0_blue == NULL)) {
        printf("At least one of LEDs is null.\n");
        return 1;
    }

    // Enable each LED
    metal_led_enable(led0_red);
    metal_led_enable(led0_green);
    metal_led_enable(led0_blue);

    // All Off
    metal_led_off(led0_red);
    metal_led_off(led0_green);
    metal_led_off(led0_blue);

    // Lets get the CPU and its interrupt
    cpu = metal_cpu_get(metal_cpu_get_current_hartid());
    if (cpu == NULL) {
        printf("CPU null.\n");
        return 2;
    }
    cpu_intr = metal_cpu_interrupt_controller(cpu);
    if (cpu_intr == NULL) {
        printf("CPU interrupt controller is null.\n");
        return 3;
    }
    metal_interrupt_init(cpu_intr);

    // display welcome banner
    display_banner();

	printf("Starting MNIST inference...\n");
	BitMnistInference(input_data_0, label_0,1);
	BitMnistInference(input_data_1, label_1,2);
	BitMnistInference(input_data_2, label_2,3);
	BitMnistInference(input_data_3, label_3,4);

    // Setup Timer and its interrupt so we can toggle LEDs on 1s cadence
    tmr_intr = metal_cpu_timer_interrupt_controller(cpu);
    if (tmr_intr == NULL) {
        printf("TIMER interrupt controller is  null.\n");
        return 4;
    }
    metal_interrupt_init(tmr_intr);
    tmr_id = metal_cpu_timer_get_interrupt_id(cpu);
    rc = metal_interrupt_register_handler(tmr_intr, tmr_id, timer_isr, cpu);
    if (rc < 0) {
        printf("TIMER interrupt handler registration failed\n");
        return (rc * -1);
    }

    // Lastly CPU interrupt
    if (metal_interrupt_enable(cpu_intr, 0) == -1) {
        printf("CPU interrupt enable failed\n");
        return 6;
    }

    // Red -> Green -> Blue, repeat
    while (1) {

        // Turn on RED
        wait_for_timer(led0_red);

        // Turn on Green
        wait_for_timer(led0_green);

        // Turn on Blue
        wait_for_timer(led0_blue);

    }

    // return
    return 0;
}
