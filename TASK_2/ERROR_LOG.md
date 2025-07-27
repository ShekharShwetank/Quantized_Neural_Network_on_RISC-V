## Troubleshooting Log: VSDSquadron PRO Edge AI Model Deployment

This document details steps for troubleshooting and successfully building a TensorFlow Lite Micro application for the SiFive FE310-G002 on the VSDSquadron PRO board. It outlines the challenges faced, learning from each obstacle, and how I ultimately solved them.

---

### Error Log and Resolutions

#### Problem 1: Initial Model's Memory Footprint: Too Large

* **The Problem Faced:** Initial Neural Network model, even after training and Python-based quantization, had a memory footprint (102.05 KB for the `.tflite` model, with an estimated 99.38 KB for quantized weights plus activations) that far exceeded the SiFive FE310-G002's 16KB Data SRAM.
* **Figured Out:** Realized that hardware memory constraints are a critical factor in embedded AI. Standard quantization (converting `float32` to `int8`) significantly reduced the model size (approximately 4x in my case), but for extremely limited SRAM, the model's architectural complexity itself needed to be reduced. The board has 32 Mbit (4 MB) of off-chip flash memory for storing the model and 16KB of Data SRAM for runtime operations.

* **Fix:**
    1.  **Reduced Model Architecture:** Iteratively scaled down the number of neurons in Neural Network's hidden layer.
        * 1st attempt at reduction still left the quantized model at **27.23 KB**, which was still too large.
        * 2nd attempt, reducing the hidden layer to **16 neurons**, finally resulted in a quantized model size of **14.79 KB**.
        * 3rd attempt, reducing the hidden layer to **8 neurons**, finally resulted in a quantized model size of **8.55 KB**.
    2.  **Verified Fit:** This `8.55 KB` model size successfully fits within the 16KB Data SRAM, making deployment on the board feasible.

#### Problem 2: `main.c` Wasn't Compiling / Received `undefined reference to 'main'`

* **The Problem Faced:** After setting up C++ application code (`main.cc`) and integrating TensorFlow Lite Micro logic, build failed with an `undefined reference to 'main'` error. This clearly indicated that `main.cc` file wasn't being compiled or linked correctly.
* **Figured Out:** The default "empty" project template in Freedom Studio's top-level `Makefile` was configured to compile `main.c`. When renamed file to `main.cc` (to work with C++ TFLM APIs), the build system wasn't automatically recognizing or compiling `main.cc`. The actual issue was rooted in `src/Makefile`, which governs the compilation of sources within the `src` directory.
* **My Fix:**
    1.  **Renamed `PROGRAM` in top-level `Makefile`:** updated `PROGRAM = empty` to `PROGRAM = main` in the top-level `Makefile` to reflect `main.cc` file.
    2.  **Modified `src/Makefile`:** I completely replaced the entire content of `TASK_2_2_EDGE_AI_SHWETANK/src/Makefile` with a more comprehensive version. This new `src/Makefile` now includes:
        * Definitions for `TFLM_BASE_DIR` and `FLATBUFFERS_BASE_DIR` to correctly locate TFLite Micro sources relative to `src/`.
        * `VPATH` directives to help `make` find source files in nested directories.
        * Explicit lists of all application (`APP_SRCS`) and TensorFlow Lite Micro core (`TFLM_MICRO_SRCS`, `TFLM_KERNELS_SRCS`, `FLATBUFFERS_SRCS`) C++ source files.
        * Generic compilation rules (like `%.o: %.cc`) for C++ files, ensuring `riscv64-unknown-elf-g++` is used.
        * An explicit rule for `main.o: main.cc` to guarantee its proper compilation as a C++ file.

#### Problem 3: Persistent `Makefile:58: *** missing separator. Stop.`

* **The Problem Faced:** After modifying `src/Makefile`, a `missing separator` error kept appearing on a specific line (line 58) within `src/Makefile`, blocking the `make` process.
* **Figured Out:** Learned that this error is a classic `Makefile` trap. It happens when a line that `make` expects to be a shell command (which *must* start with a literal tab character) is instead indented with spaces, or contains an invisible character.
* **Fix:**
    1.  **Meticulous Tab Correction:** Manually went through `src/Makefile` line by line.
    2.  **Replaced all leading spaces on every single command line with a single, literal tab character.** This required extreme precision and patience to ensure every command was correctly tab-indented.

#### Problem 4: Still Couldn't Find Headers: `fatal error: tensorflow/lite/micro/micro_mutable_op_resolver.h: No such file or directory` (and many subsequent similar issues, then currently at `tensorflow/lite/core/c/common.h` issue)

* **The Problem Faced:** Even after fixing the `missing separator` and updating `src/Makefile` for C++ compilation, the build continued to fail with "No such file or directory" errors for TFLite Micro headers, first for `micro_mutable_op_resolver.h` and many subsequent similar issues, then currently failing for internal TFLM headers like `tensorflow/lite/core/c/common.h`. While the compilation command showed the include paths were being passed, the compiler still couldn't locate these files.
* **Figured Out:** This was a nuanced "header file not found" issue. The problem wasn't a missing `-I` flag, but rather that the *base* include path I was providing didn't match TFLite Micro's internal expectations. TFLite Micro's internal headers are structured to be included *relative to the root of the TensorFlow repository itself* (i.e., the directory *containing* the `tensorflow` folder). previous `-I` paths were either too specific or didn't provide this top-level context correctly.
* **Fix:**
    1.  **Refined Top-Level Makefile Include Paths:** Adjusted the `RISCV_CFLAGS` and `RISCV_CXXFLAGS` variables in the **top-level `Makefile`**.
    2.  Updated `TFLM_BASE_DIR` to point precisely to `$(abspath $(SRC_DIR)/tensorflow_lite_micro_src)`, which is the directory that directly contains the `tensorflow` folder in copied TFLite Micro source tree.
    3.  Ensured that **only** `-I$(TFLM_BASE_DIR)` and `-I$(FLATBUFFERS_INCLUDE_DIR)` were included for TFLM-related headers in the compiler flags. This ensures the compiler, when seeing `#include <tensorflow/...>`, correctly looks within `tensorflow_lite_micro_src/tensorflow` hierarchy. This finally allowed the compiler to find the nested TFLite Micro headers.
    4. Working to resolve the current `tensorflow/lite/core/c/common.h` issue. The problem is that the .h files being refernced by some of the other files aren't present in the correct path or don't exist( even in the original repository).
---

**Current Status:**

As of now, the `missing separator` error is resolved, and the compiler is successfully finding the TFLite Micro headers and compiling `main.cc`. The warnings about "overriding recipe" are still present but are not build-breaking. The next steps will involve addressing any linker errors ("undefined reference to..."), which will indicate that specific TFLite Micro object files or additional standard libraries need to be linked into the final executable.

Working to resolve the current `tensorflow/lite/core/c/common.h` issue. The problem is that the .h files being refernced by some of the other files aren't present in the correct path or don't exist( even in the original repository).

