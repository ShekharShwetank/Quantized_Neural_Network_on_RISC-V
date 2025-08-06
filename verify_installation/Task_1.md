# VSDSquadron PRO Board Specifications

## 1. Board Overview

The **VSDSquadron PRO** board is a development platform for IoT and edge computing, powered by the **SiFive FE310-G002 RISC-V SoC**.

| Feature         | Specification                                |
| :-------------- | :------------------------------------------- |
| **Form Factor** | 84.00 x 52.00 mm (Max Height Top: 8mm, Bottom: 1mm) |
| **I/O Voltage** | 3.3V                                         |
| **Input Voltage** | 5V (Nominal)                                 |
| **Operating Temperature** | 20°C to 35°C (68°F to 95°F)                 |
| **USB Interface** | USB-C Type (via FT2232 USB-to-Serial Converter) |
| **Crystal Oscillators** | On-board 12MHz, 16MHz                        |

---

## 2. SiFive FE310-G002 SoC

The core of the VSDSquadron PRO, the **SiFive FE310-G002**, is a 32-bit RISC-V microcontroller designed for efficient processing.

### 2.1 CPU Core (E31 Core Complex)

| Feature                 | Specification                                    |
| :---------------------- | :----------------------------------------------- |
| **Architecture** | RV32IMAC ISA (32-bit)                            |
| **Frequency** | Up to 320MHz                                     |
| **Performance** | 1.61 DMIPS/MHz, 2.73 Coremark/MHz                |
| **Hardware Accelerators** | Integer Multiply/Divide (8-bit/cycle multiply, 1-bit/cycle divide) |
| **Branch Predictor** | 40 BTB entries, 128 BHT entries, 2-entry RAS     |

### 2.2 Memory Subsystem

| Memory Type               | Specification                                |
| :------------------------ | :------------------------------------------- |
| **Instruction Cache (L1)** | 16KB, 2-way set associative, 32-byte lines   |
| **Data SRAM (L1)** | 16KB (DTIM), 2-cycle access latency for full words |
| **Mask ROM (MROM)** | 8KB (Boot code, platform config, debug routines) |
| **OTP Program Memory** | 8KB, in-circuit programmable                 |
| **Off-Chip SPI Flash** | 32 Mbit (ISSI SPI Flash) on board; QSPI 0 supports up to 512 MiB |

### 2.3 Peripherals and Interfaces

The FE310-G002 provides extensive peripheral connectivity.

| Interface      | Details                                          |
| :------------- | :----------------------------------------------- |
| **GPIO** | 19 Digital I/O pins (on board), 32 on chip. Configurable as input/output, pull-ups, drive strengths, output inversion. |
| **UART** | 2 instances (UART0, UART1). 8-entry TX/RX FIFO.  |
| **I2C** | 1 instance (I2C0).                               |
| **QSPI** | Dedicated flash interface (QSPI_DQ_x, QSPI_CS, QSPI_SCK). Two additional QSPI controllers in GPIO block. |
| **PWM** | 3 independent controllers: PWM0 (8-bit, 4 comparators), PWM1 (16-bit, 4 comparators), PWM2 (16-bit, 4 comparators). 9 PWM pins on board. |
| **External Interrupts** | 19 pins                                          |
| **External Wakeup** | 1 pin (AON_PMU_DWAKEUP_N)                      |

---

## 3. Debugging Features

The FE310-G002 supports robust debugging.

| Feature        | Details                                          |
| :------------- | :----------------------------------------------- |
| **JTAG** | 4-wire IEEE 1149.1 compliant. Max jtag_TCK frequency is part specific. |
| **Debug Module** | 8 programmable hardware breakpoints. Implements BYPASS and IDCODE (0x20000913) instructions. Access via JTAG DEBUG instruction using a 2-bit opcode, 7-bit address, and 32-bit data field. |

---

## 4. Power Management

| Feature               | Details                                      |
| :-------------------- | :------------------------------------------- |
| **Power Domains** | Multiple isolated power domains              |
| **Standby Mode** | Low-power standby mode                       |
| **Supply Voltages (Chip)** | Core Logic: 1.8V, I/O Pads & AON Block: 3.3V |
| **Typical Current (Chip)** | IVDD: 8mA (16MHz), 16mA (250MHz). VDD: 8mA (16MHz), 150mA (250MHz). |
