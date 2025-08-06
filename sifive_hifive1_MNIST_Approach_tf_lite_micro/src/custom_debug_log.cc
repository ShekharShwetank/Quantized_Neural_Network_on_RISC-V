#include <metal/uart.h> // For metal_uart_putc and metal_uart_txready
#include <stdio.h> // For vsnprintf
#include <stdarg.h> // For va_list, va_start, va_end

extern struct metal_uart *uart0; // Declare the global UART handle from main.cc

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// This is a minimal implementation for DebugLog, typically used by TFLite Micro
// when MicroPrintf is not available or explicitly redirected.
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

// This function is expected by MicroVsnprintf for formatted output.
// It uses standard vsnprintf for formatting and then DebugLog for actual output.
int DebugVsnprintf(char* buffer, size_t buf_size, const char* format, va_list vlist) {
    int result = vsnprintf(buffer, buf_size, format, vlist);
    DebugLog(buffer); // Output the formatted string
    return result;
}

#ifdef __cplusplus
}
#endif // __cplusplus
