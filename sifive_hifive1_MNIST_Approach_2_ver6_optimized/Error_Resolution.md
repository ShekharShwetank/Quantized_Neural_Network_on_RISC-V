# Error Resolution Log: Quantized MNIST Inference

This document meticulously chronicles the challenges encountered and the solutions implemented during the process of porting a TensorFlow Lite quantized MNIST handwritten digit classification model to the `VSDSquadron PRO development board (SiFive FE310-G002 RISC-V SoC)`. The project utilizes a custom BitNet-like inference engine running on the `SiFive FE310-G002 RISC-V` microcontroller.

## Project Overview

The primary goal was to successfully execute a quantized neural network model on the target hardware. This involved transforming a TensorFlow Lite (`.tflite`) model into C-compatible data structures, integrating these into a bare-metal C application, and ensuring numerical correctness and functional operation on the SiFive FE310-G002.

## Final Outcome

The project successfully achieved the deployment of a quantized MNIST classification model onto the SiFive FE310-G002 on the VSDSquadron PRO board. The resolution process highlighted the importance of precise numerical implementation details in embedded quantized inference engines, particularly concerning sign extension, scaling, and robust integer arithmetic, alongside correct project configuration and tooling usage.

## Environment

  * **Development Environment:** Freedom Studio 3.1.1
  * **Target Board:** VSDSquadron PRO (SiFive FE310-G002 RISC-V SoC)
  * **Toolchain:** SiFive RISC-V Metal GNU GCC Toolchain
  * **Inference Engine Core:** Custom C functions (like: `processfclayer`, `ReLUNorm`)
  * **Model Source:** TensorFlow Lite (`mnist_quantized_model.tflite`) derived from a Keras model, quantized to 8-bit integers.
  * **Python Scripts (Jupiter NB):** Used for `.tflite` model training, generation & parsing and C array generation.

## Detailed Error Log and Resolutions

Section: issue diagnosis, and the precise steps taken for resolution.

### Issue 1: Incorrect Function Section Attribute

  * **Symptom:** Compiler warnings or errors indicating that the `processfclayer` function (executable code) was being placed into a read-only data section (`.srodata`).
    ```c
    void processfclayer(int8_t *,  const uint32_t *, int32_t, uint32_t, uint32_t, int32_t *) __attribute__((section(".srodata"))) __attribute__((used));
    ```
  * **Diagnosis:** The `__attribute__((section(".srodata")))` compiler directive was erroneously applied to the `processfclayer` function declaration. Functions, as executable code, belong in text or code segments, not data segments. Attempting to execute from a data segment can lead to segmentation faults or undefined behavior on embedded systems with Memory Protection Units (MPU/PMP).
  * **Resolution Steps:**
    1.  **File:** `main.c`
    2.  **Modification:** Removed the `__attribute__((section(".srodata"))) __attribute__((used));` portion from the `processfclayer` function declaration.
        ```c
        // Before:
        // void processfclayer(int8_t *,  const uint32_t *, int32_t, uint32_t, uint32_t, int32_t *) __attribute__((section(".srodata"))) __attribute__((used));
        // After:
        void processfclayer(int8_t *,  const uint32_t *, int32_t, uint32_t, uint32_t, int32_t *);
        ```
  * **Verification:** The specific attribute-related compilation warnings/errors were resolved.

### Issue 2: Undeclared Model Parameters and Mismatched Input Data

  * **Symptom:** Compiler errors indicating "undeclared" identifiers for model parameters such as `MAX_N_ACTIVATIONS`, `L1_weights`, `L1_bitperweight`, `L1_incoming_weights`, `L1_outgoing_weights`, etc., within the `BitMnistInference` function in `main.c`. Additionally, the hardcoded `input_data_X` arrays in `main.c` were of size `[256]`, while the flattened MNIST input image size is `784`.
  * **Diagnosis:** The `main.c` file was attempting to use neural network layer parameters without their proper C definitions. These parameters were expected to be generated from the quantized TensorFlow Lite model. The initial input data samples were also incorrectly sized and possibly not correctly quantized for the model's input expectations. The `mnist_baseline_model.ipynb` indicated a 2-layer dense network after flattening (784 -\> 8 -\> 10 classes).
  * **Resolution Steps:**
    1.  **File:** `main.c`
    2.  **Modification (Constants and Layer Count):**
          * Updated `#define MAX_N_ACTIVATIONS 784` to match the largest layer input size (28x28 pixels flattened).
          * Modified the `BitMnistInference` function calls to explicitly align with a 2-layer network (two calls to `processfclayer` and `ReLUNorm` for `L1` and `L2`).
          * Removed hardcoded placeholder definitions for `L*_weights` and associated parameters (as these would be generated).
          * Removed hardcoded `input_data_X` and `label_X` arrays.
  * **Verification:** The compiler errors related to undeclared parameters were expected to persist until the generation step was complete. This set the C code structure for correct parameter and input usage.

### Issue 3: Python Script `AttributeError: 'Interpreter' object has no attribute 'get_subgraph_details'`

  * **Symptom:** When attempting to run the Python script `generate_c_model_params.py` (which was intended to extract model parameters), a `AttributeError` occurred:
    ```
    AttributeError: 'Interpreter' object has no attribute 'get_subgraph_details'
    ```
  * **Diagnosis:** The specific TensorFlow Lite API call `interpreter.get_subgraph_details()` was not available or its path had changed in the user's TensorFlow version (2.15.0), preventing programmatic inspection of the `.tflite` model's operator graph.
  * **Resolution Steps:**
    1.  **File:** `generate_c_model_params.py`.
    2.  **Modification:** Changed the method used to retrieve operator details from the interpreter.
        ```python
        # Before (attempted or assumed path):
        # op_details = interpreter.get_subgraph_details(0)['operators']
        # After:
        ops_details_raw = interpreter._get_ops_details()
        ```
  * **Verification:** This allowed the script to proceed and access a list of raw operator details.

### Issue 4: Python Script `AttributeError: 'Interpreter' object has no attribute '_get_builtin_op_name'`

  * **Symptom:** After resolving the previous `AttributeError`, another `AttributeError` occurred, indicating the absence of `_get_builtin_op_name`:
    ```
    AttributeError: 'Interpreter' object has no attribute '_get_builtin_op_name'
    ```
  * **Diagnosis:** The internal method `interpreter._get_builtin_op_name()` (used to get human-readable names from `builtin_code`) was also unavailable or had changed in the TensorFlow version, hindering the identification of `FULLY_CONNECTED` layers.
  * **Resolution Steps:**
    1.  **File:** `generate_c_model_params.py`.
    2.  **Modification:** Updated the script to directly compare the `builtin_code` integer value within the `op_detail` dictionary against the known integer code for `FULLY_CONNECTED` (`9`), bypassing the need for the problematic method.
        ```python
        # Inside the loop for op_detail in ops_details_raw:
        # ...
        # Before:
        # op_name = interpreter._get_builtin_op_name(op_detail['builtin_code'])
        # if op_name == 'FULLY_CONNECTED':
        # After:
        current_builtin_code = op_detail.get('builtin_code') # Safely get the code
        if current_builtin_code == 9: # BuiltinOperator.FULLY_CONNECTED has builtin_code 9
            # ... process FULLY_CONNECTED layer ...
        ```
  * **Verification:** The script could now correctly identify `FULLY_CONNECTED` operations.

### Issue 5: Python Script `ModuleNotFoundError: No module named 'tensorflow.lite.python.enums'`

  * **Symptom:** The script failed with:
    ```
    ModuleNotFoundError: No module named 'tensorflow.lite.python.enums'
    ```
  * **Diagnosis:** The `BuiltinOperator` enum, which provides a symbolic constant for `FULLY_CONNECTED`, could not be imported from its specified path in the TensorFlow version.
  * **Resolution Steps:**
    1.  **File:** `generate_c_model_params.py`.
    2.  **Modification:** Removed the import statement for `BuiltinOperator` and relied solely on the direct integer comparison `current_builtin_code == 9` for operator identification, which was already implemented.
  * **Verification:** This resolved the `ModuleNotFoundError`.

### Issue 6: Python Script `KeyError: 'builtin_code'`

  * **Symptom:** The script threw a `KeyError` when trying to access `'builtin_code'` from an `op_detail` dictionary:
    ```
    KeyError: 'builtin_code'
    ```
  * **Diagnosis:** The dictionaries returned by `interpreter._get_ops_details()` (which replaced `get_subgraph_details` due to API changes) did not always contain the `'builtin_code'` key directly in that specific TensorFlow Lite version. They might be summary dictionaries.
  * **Resolution Steps:**
    1.  **File:** `generate_c_model_params.py`.
    2.  **Modification:** Implemented safer dictionary access using `.get()` for `op_name` and `builtin_code` to prevent `KeyError` if the key was absent. The logic for identifying `FULLY_CONNECTED` layers then checked both `op_name` and `builtin_code`.
  * **Verification:** This improved script robustness and allowed it to parse operator details without crashing on missing keys.

### Issue 7: Python Script `TypeError: Interpreter.get_tensor_details() takes 1 positional argument but 2 were given`

  * **Symptom:** The script crashed with:
    ```
    TypeError: Interpreter.get_tensor_details() takes 1 positional argument but 2 were given
    ```
  * **Diagnosis:** This error indicated that `interpreter.get_tensor_details()` was receiving two arguments instead of the expected single integer tensor index. This was due to an unexpected format of the tensor indices from `op_detail['inputs']` or `op_detail['outputs']`, possibly being tuples/lists that were implicitly unpacked.
  * **Resolution Steps:**
    1.  **File:** `generate_c_model_params.py`.
    2.  **Modification:** Changed the approach for retrieving `tensor_details` from the interpreter. Instead of direct indexing, the script now iterates through `interpreter.get_tensor_details()` and finds the tensor by matching its `index`. For retrieving tensor data, `interpreter.tensor(tensor_idx)()` was used, which is the correct way to get the actual NumPy array of data.
        ```python
        # Instead of interpreter.get_tensor_details(input_tensor_idx):
        input_tensor = next(t for t in tensor_details if t['index'] == input_tensor_idx)
        # ... similarly for weights_tensor, output_tensor
        # For actual tensor data:
        weights_data = interpreter.tensor(weights_tensor_idx)()
        biases_data = interpreter.tensor(biases_tensor_idx)() if biases_tensor_idx != -1 else None
        ```
  * **Verification:** This resolved the `TypeError` and enabled successful extraction of tensor details and data.

### Issue 8: Program Hangs After "Starting MNIST inference..."

  * **Symptom:** The compiled application successfully loaded onto the VSDSquadron PRO board and printed initial messages like the SiFive banner and "Starting MNIST inference...", but then ceased execution without further output, indicating a hang or unhandled exception.
  * **Diagnosis:** This was a critical runtime failure stemming from fundamental numerical errors within the inference engine:
    1.  **Incorrect 4-bit Weight Sign Extension:** The most severe issue. The `processfclayer` function in `app_inference.h` (or `BitNetMCU_inference.h` from the broader project context) contained a logical flaw in how it unpacked and interpreted 4-bit signed symmetric weights from `uint32_t` chunks. A logical right shift (`>>`) on an unsigned type fills with zeros, incorrectly transforming negative 4-bit values (which have their MSB set) into large positive numbers when promoted to `int32_t`. This corrupted all subsequent calculations. The model was confirmed to use 4-bit symmetric weights.
    2.  **`ReLUNorm` Scaling Instability:** The `ReLUNorm` function contained a brittle implementation for calculating the scaling `shift` and `rounding` values. Specifically, `1 << (shift - 1)` could result in undefined behavior if `shift` evaluated to 0, which could cause a crash or incorrect scaling leading to all-zero activations.
    3.  **`MAX_N_ACTIVATIONS` Discrepancy:** While the Python model implied a maximum activation count of 784 (for the flattened input), the `BitNetMCU_model_12k.h` specified `MAX_N_ACTIVATIONS 64`, which affected buffer sizing and potentially indexing.
  * **Resolution Steps:**
    1.  **File:** `app_inference.h`
    2.  **Modification (4-bit Weight Extraction Fix):** Corrected the `else if (bits_per_weight == 4)` block in `processfclayer` to ensure proper 4-bit signed value extraction and sign-extension.
        ```c
        // Inside processfclayer, for bits_per_weight == 4:
        // ...
        uint8_t nibble = (uint8_t)((weightChunk >> 28) & 0xF); // Extract 4-bit value to LSB of a byte
        int32_t weight;
        if (nibble & 0x8) { // Check the sign bit (MSB of the 4-bit nibble)
            weight = (int32_t)(nibble | 0xFFFFFFF0); // Sign extend to 32-bit by OR-ing with all higher bits set
        } else {
            weight = (int32_t)nibble; // Positive, direct cast
        }
        sum += in * weight;
        weightChunk <<= 4; // Shift for next nibble
        // ...
        ```
    3.  **File:** `app_inference.h`
    4.  **Modification (`ReLUNorm` Robustness):** Rewrote the `ReLUNorm` function's scaling logic to be entirely integer-based and robust to edge cases, preventing undefined behavior.
        ```c
        // Inside ReLUNorm:
        int32_t max_val = 0; // Initialize for ReLU (non-negative max)
        // ... (find max_val loop)
        if (max_val <= 0) { // All inputs are non-positive
            for (uint32_t i = 0; i < n_input; i++) output[i] = 0;
            return max_pos;
        }
        uint32_t shift = 0;
        uint32_t temp_max_val = (uint32_t)max_val;
        while (temp_max_val > 127) { // Calculate shift to fit in [0, 127]
            temp_max_val >>= 1;
            shift++;
        }
        int32_t rounding = (shift > 0) ? (1 << (shift - 1)) : 0; // Robust rounding calculation
        // ... (apply ReLU, scaling, and clipping)
        ```
    5.  **File:** `main.c`
    6.  **Modification (`MAX_N_ACTIVATIONS`):** Corrected `#define MAX_N_ACTIVATIONS 64` to align with the model's actual intermediate layer sizes, as defined in the associated model header (`mnist_model_params.h` and indirectly from `BitNetMCU_model_12k.h`).

* **Verification:** After applying these critical fixes, the program successfully executed on the VSDSquadron PRO board, performed inferences for all sample inputs, and printed the expected output, confirming functional model execution.
