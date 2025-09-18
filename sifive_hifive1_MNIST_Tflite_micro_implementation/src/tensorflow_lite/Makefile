# Copyright 2019 SiFive, Inc #
# SPDX-License-Identifier: Apache-2.0 #
# src/tensorflow_lite/Makefile #
# Makefile to build TensorFlow Lite Micro as a static library #

TFLM_SRCS := $(shell find . -type f \( -name '*.cc' -o -name '*.c' \))
TFLM_OBJS := $(TFLM_SRCS:.cc=.o)
TFLM_OBJS := $(TFLM_OBJS:.c=.o)

CXXFLAGS += -I. -Ithird_party/flatbuffers/include -Ithird_party/gemmlowp -Ithird_party/kissfft -Ithird_party/ruy

all: libtflm.a

libtflm.a: $(TFLM_OBJS)
	ar rcs $@ $^

%.o: %.cc
	riscv64-unknown-elf-g++ $(CXXFLAGS) -c $< -o $@

%.o: %.c
	riscv64-unknown-elf-g++ $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TFLM_OBJS) libtflm.a