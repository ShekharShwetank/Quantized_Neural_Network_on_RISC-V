# Define the project name and target, passed from the parent Makefile
PROGRAM ?= main # Default to 'main' if not passed
TARGET ?= sifive-hifive1 # Default to 'sifive-hifive1' if not passed

# Define the base directory for TFLite Micro sources relative to SRC_DIR (current directory)
# Ensure 'tensorflow_lite_micro_src' matches the folder name you created
TFLM_BASE_DIR = tensorflow_lite_micro_src/tensorflow/lite
FLATBUFFERS_BASE_DIR = tensorflow_lite_micro_src/third_party/flatbuffers/src

# Vpath tells make where to find prerequisites (source files) that aren't in the current directory
VPATH := $(TFLM_BASE_DIR)/micro \
         $(TFLM_BASE_DIR)/kernels \
         $(FLATBUFFERS_BASE_DIR)

# Your application sources (main.cc)
APP_SRCS = main.cc

# List TensorFlow Lite Micro source files (relative to their own subdirectories, for VPATH)
TFLM_MICRO_SRCS = \
    micro_interpreter.cc \
    micro_error_reporter.cc \
    micro_log.cc \
    system_setup.cc \
    all_ops_resolver.cc \
    memory_helpers.cc \
    micro_allocator.cc \
    micro_resource_variable.cc \
    micro_time.cc \
    micro_utils.cc

TFLM_KERNELS_SRCS = \
    fully_connected.cc \
    softmax.cc \
    quantize.cc \
    dequantize.cc \
    reshape.cc

# Flatbuffers utility source
FLATBUFFERS_SRCS = \
    util.cc

# Combine all source files into one list.
# These paths are relative to VPATH entries or the current directory (src/).
ALL_SRCS = $(APP_SRCS) $(TFLM_MICRO_SRCS) $(TFLM_KERNELS_SRCS) $(FLATBUFFERS_SRCS)

# Convert source files to object files.
# This will create .o files in the current directory (src/)
OBJS = $(patsubst %.cc,%.o,$(filter %.cc,$(ALL_SRCS))) \
       $(patsubst %.c,%.o,$(filter %.c,$(ALL_SRCS))) \
       $(patsubst %.S,%.o,$(filter %.S,$(ALL_SRCS)))

# Define the default target for the src/Makefile
# This target should depend on all object files
# It then calls the linker to create the final executable.
$(PROGRAM): $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) $(LDLIBS) -o $@

# Generic rule to compile .c files (if any C sources are present in ALL_SRCS)
%.o: %.c
	$(CC) $(CFLAGS) $(XCFLAGS) -c $< -o $@

# Generic rule to compile .cc files (C++)
# This rule relies on VPATH to find the source files
%.o: %.cc
	@echo "Compiling C++ file: $<"
	$(CXX) $(CXXFLAGS) $(XCFLAGS) -c $< -o $@

# Rule to compile main.cc explicitly (optional, but ensures it's handled as C++)
main.o: main.cc
	@echo "Compiling main.cc"
	$(CXX) $(CXXFLAGS) $(XCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	$(RM) -f $(OBJS) $(PROGRAM) $(PROGRAM).map