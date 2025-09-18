// C:/VSD_Sqd_Project/TASK_2_2_EDGE_AI_SHWETANK/src/tensorflow_lite_micro_src/tensorflow/lite/micro/micro_log.cc
/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This file is the default implementation for MicroPrintf and other logging
// facades in TensorFlow Lite Micro. It uses the DebugLog function to
// actually print the formatted strings. To configure logging for your
// platform, you should provide an implementation of DebugLog.

#include <metal/uart.h> // Include metal UART for actual hardware access
#include <stdio.h> // For vsnprintf
#include <stdarg.h> // For va_list, va_start, va_end

// Declare the global UART handle from main.cc
// This is a global symbol that the linker will resolve.
extern struct metal_uart *uart0;

// Maximum size of a single formatted log message.
// This must be large enough to hold typical messages.
// Adjust as needed, considering memory constraints.
#ifndef TF_LITE_STRIP_ERROR_STRINGS
// Maximum size of a single formatted log message.
// This must be large enough to hold typical messages.
// Adjust as needed, considering memory constraints.
constexpr int kMaxLogBufferSize = 256; // Corrected declaration and a more typical buffer size

// A buffer to format messages into. This is a global to avoid stack usage.
char log_buffer[kMaxLogBufferSize];
#endif // TF_LITE_STRIP_ERROR_STRINGS

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// Implement DebugLog using metal_uart for direct character output.
// This function is typically called by DebugVsnprintf after formatting.
void DebugLog(const char* s) {
  if (uart0 != NULL) {
    while (*s != '\0') {
      // Wait until TX is ready
      while (!metal_uart_txready(uart0));
      metal_uart_putc(uart0, *s);
      s++;
    }
  }
}

// Implement DebugVsnprintf using standard vsnprintf and then our DebugLog.
// This is the core formatting function that MicroPrintf eventually calls.
int DebugVsnprintf(char* buffer, size_t buf_size, const char* format, va_list vlist) {
    // Use the standard C library's vsnprintf to format the string.
    // This function writes at most buf_size characters (including null terminator) to buffer.
    int result = vsnprintf(buffer, buf_size, format, vlist);
    return result;
}

#ifdef __cplusplus
}
#endif // __cplusplus

namespace tflite {
namespace micro {

// This is the implementation of MicroPrintf that calls the platform's DebugVsnprintf.
void MicroPrintf(const char* format, ...) {
#ifndef TF_LITE_STRIP_ERROR_STRINGS
  va_list args;
  va_start(args, format);
  // Use DebugVsnprintf to format the string into the global buffer.
  int chars_written = DebugVsnprintf(log_buffer, kMaxLogBufferSize, format, args);
  va_end(args);

  // Print the formatted string using the platform's DebugLog.
  DebugLog(log_buffer);

  // If the message was truncated, add an indicator.
  if (chars_written > kMaxLogBufferSize -1) {
      DebugLog("...\n"); // Indicate truncation
  } else {
      DebugLog("\n"); // Add newline for typical logging behavior
  }
#endif // TF_LITE_STRIP_ERROR_STRINGS
}

}  // namespace micro
}  // namespace tflite
