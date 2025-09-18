# Minimal TensorFlow Lite Micro Makefile for MNIST
# Only includes essential components
# src/Makefile

PROGRAM ?= sifive-welcome

# --- Application Source ---
CXX_SRCS = $(wildcard *.cc)

# --- TensorFlow Lite Micro - CORE ONLY ---
TFLM_DIR = tensorflow_lite

# Core TFLM Framework (Essential)
TFLM_CORE_SRCS = \
    $(TFLM_DIR)/tensorflow/lite/micro/micro_interpreter.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/micro_interpreter_graph.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/micro_interpreter_context.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/micro_allocation_info.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/flatbuffer_utils.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/micro_allocator.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/micro_context.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/micro_error_reporter.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/micro_op_resolver.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/micro_log.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/micro_time.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/memory_helpers.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/micro_utils.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/system_setup.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/micro_resource_variable.cc

# Memory Management (Essential)
TFLM_MEMORY_SRCS = \
    $(TFLM_DIR)/tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/memory_planner/linear_memory_planner.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.cc

# Core API (Essential)
TFLM_API_SRCS = \
    $(TFLM_DIR)/tensorflow/lite/core/api/flatbuffer_conversions.cc \
    $(TFLM_DIR)/tensorflow/lite/core/c/common.cc

# MLIR/core helper used by ConvertTensorType
TFLM_CORE_API_SRCS = \
    $(TFLM_DIR)/tensorflow/compiler/mlir/lite/core/api/error_reporter.cc

# Schema Support (Essential)
TFLM_SCHEMA_SRCS = \
    $(TFLM_DIR)/tensorflow/compiler/mlir/lite/schema/schema_utils.cc

# ONLY Required Kernels for MNIST
TFLM_KERNEL_SRCS = \
    $(TFLM_DIR)/tensorflow/lite/micro/kernels/fully_connected.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/kernels/fully_connected_common.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/kernels/softmax.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/kernels/softmax_common.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/kernels/reshape.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/kernels/reshape_common.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/kernels/quantize.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/kernels/quantize_common.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/kernels/dequantize.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/kernels/dequantize_common.cc

# Add LeakyRelu kernel (used by your model)
TFLM_KERNEL_SRCS += \
    $(TFLM_DIR)/tensorflow/lite/micro/kernels/leaky_relu.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/kernels/leaky_relu_common.cc

# Essential Kernel Utilities
TFLM_KERNEL_UTILS = \
    $(TFLM_DIR)/tensorflow/lite/kernels/kernel_util.cc \
    $(TFLM_DIR)/tensorflow/lite/kernels/internal/quantization_util.cc \
    $(TFLM_DIR)/tensorflow/lite/kernels/internal/common.cc \
    $(TFLM_DIR)/tensorflow/lite/kernels/internal/portable_tensor_utils.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/kernels/kernel_util.cc

# Flatbuffer-to-micro bridge helpers
TFLM_FLATBUFFER_BRIDGE_SRCS = \
    $(TFLM_DIR)/tensorflow/lite/micro/flatbuffer_conversions_bridge.cc

# Platform Support (RISC-V)
TFLM_PLATFORM_SRCS = \
    $(TFLM_DIR)/tensorflow/lite/micro/riscv32_generic/debug_log.cc

# Third-party - ONLY what's actually used
TFLM_THIRD_PARTY_SRCS = \
    $(TFLM_DIR)/third_party/eyalroz_printf/src/printf/printf.c

# Combine all minimal sources
SRCS = $(CXX_SRCS) $(wildcard *.c) \
       $(TFLM_CORE_SRCS) \
       $(TFLM_MEMORY_SRCS) \
       $(TFLM_API_SRCS) \
       $(TFLM_CORE_API_SRCS) \
       $(TFLM_FLATBUFFER_BRIDGE_SRCS) \
       $(TFLM_SCHEMA_SRCS) \
       $(TFLM_KERNEL_SRCS) \
       $(TFLM_KERNEL_UTILS) \
       $(TFLM_PLATFORM_SRCS) \
       $(TFLM_THIRD_PARTY_SRCS)

# Generate object files
OBJS := $(patsubst %.c, %.o, $(filter %.c,$(SRCS))) \
	$(patsubst %.cc, %.o, $(filter %.cc,$(SRCS)))

# Remove duplicates
uniq = $(if $(1),$(firstword $(1)) $(call uniq,$(filter-out $(firstword $(1)),$(1))))
OBJS := $(call uniq,$(OBJS))

# --- Include Paths - Minimal ---
INCLUDES = \
	-I. \
	-I$(TFLM_DIR) \
	-I$(TFLM_DIR)/third_party/flatbuffers/include \
	-I$(TFLM_DIR)/third_party/eyalroz_printf/src \
	-I$(TFLM_DIR)/third_party/gemmlowp

# Combine flags
FINAL_CXXFLAGS = $(CXXFLAGS) $(INCLUDES)

# --- Build Rules ---
$(PROGRAM): $(OBJS)
	$(CXX) $(LDFLAGS) $(OBJS) $(LDLIBS) -o $@

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cc
	$(CXX) $(FINAL_CXXFLAGS) -c $< -o $@

clean:
	rm -f $(PROGRAM) $(OBJS)

# Debug: Print what files are being compiled
print-sources:
	@echo "=== Core Framework ==="
	@echo $(TFLM_CORE_SRCS)
	@echo "=== Kernels ==="
	@echo $(TFLM_KERNEL_SRCS)
	@echo "=== Total Object Files ==="
	@echo $(OBJS) | wc -w