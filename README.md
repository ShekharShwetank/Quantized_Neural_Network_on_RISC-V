# VSDSquadron PRO Board Documentation

This document outlines the hardware setup, software installation, basic program validation, and key specifications for the VSDSquadron PRO Board.

-----

## I. Hardware Setup & Software Installation

### RISC-V Toolchain and Required Software

Install the necessary drivers and development environment for the VSDSquadron PRO board.

#### Zadig Driver Installation

1.  **Download Zadig:** Obtain the `zadig.exe` utility.
2.  **Connect Board:** Plug the VSDSquadron PRO board into the PC.
3.  **Launch Zadig:** Run `zadig.exe`.
4.  **Select Device:** Go to `Options > List All Devices` and select `Dual RS-232-HS (Interface 0)`.
5.  **Install Driver:** Choose `libusb-win32 (v1.4.0.0)` from the driver list and click `Replace Driver`. Confirm the installation.

#### Freedom Studio Setup

1.  **Download Freedom Studio:** Download the `VSDSquadronPRO.tar` file for VSDSquadron PRO.
2.  **Extract:** Extract the contents of the `tar.gz` file to a preferred location.
3.  **Launch Freedom Studio:**
      * **Windows:** Navigate to the extracted `FreedomStudio` folder and run `FreedomStudio.exe`.
-----

## II. Basic Program Upload & Validation

### Verified Output
![Verified Output](TASK_1/T1_SUCCESS.png)
#### VIDEO:
https://github.com/ShekharShwetank/VSDSquadron_Pro_Edge_AI_Research_Internship/raw/master/TASK_1/T1_success_board_leds.mp4

### Testing 'sifive-welcome' Program

This section details the procedure for uploading and validating the `sifive-welcome` example program.

1.  **Open Freedom Studio:** Launch the Freedom Studio IDE.
2.  **Validation Software Project:** `SiFiveTools > Create a Software Example Project > Create a new Validation Software Project > Select SDK + Give target name > Select Example Program: "sifive-welcome" > check: 1. Build the project 2. Create a debug launch configuration for [select "OPENOCD" in the dropdown menu] > Finish`
4.  **Connect Board:** Ensure the VSDSquadron PRO board is connected to the PC via USB.
5.  **Program Board:**
        In the Debug Configuration window:
      * Select `Debug Confgurations > OpenOCD > Debug > Run > ”You have an active OpenOCD debug launch. Would you like to terminate that one and continue this one?” - Yes`.
7.  **Program Execution:** The program will upload and execute.
8.  **Verify Output:**
      * Observe the `VSD Squadron PRO` serial port output in a terminal (Freedom Studio's internal console).
-----

## III. Board Specifications

The VSDSquadron PRO board is powered by the SiFive FE310-G002 RISC-V SoC.

### FE310-G002 SiFive RISC-V SoC

![TOP_LVL_BLK_DIAG](TASK_1/FE310_G002_Top_Level_Block_Diagram.png)

  * **Core:** SiFive E31 (RV32IMAC)
      * **Architecture:** 32-bit RISC-V
      * **Frequency:** Up to 320 MHz
      * **Cache:** 16KB L1 Instruction Cache
      * **Pipelining:** 4-stage pipeline
      * **Multiplier/Divider:** Single-cycle hardware multiplier, hardware divider
  * **Debug:** JTAG Debug Interface (2-bit opcode, 7-bit debug module address, 32-bit data field)

![BLK_DIAG](TASK_1/BLOCK_DIAG.png)
### Memory & Storage

  * **Data SRAM:** 16KB
  * **Flash Memory:** 128Mbit (16MB) QSPI Flash

### Peripherals & I/O
![PINOUT](TASK_1/FE310_G002_Pinout.png)

  * **USB:** USB to Serial converter (CP2102N)
  * **Clock:** On-board 16 MHz Crystal Oscillator
  * **LEDs:**
      * 3 User LEDs
      * 1 Debug LED
      * 1 Power LED
  * **Buttons:**
      * 1 Reset Button
      * 1 IO Button
  * **IO & Protocols:**
      * 48-lead 6x6 QFN package
      * 19 Digital IO pins and 9 PWM pins
      * 2 UART and 1 I2C
      * Dedicated quad-SPI (QSPI) flash interface
      * 32 Mbit Off-Chip (ISSI SPI Flash)
      * USB-C type for Program, JTAG Debug, and Serial Communication
  * **Form Factor**
      * L x B: 84.00 x 52.00 mm
      * Height: 8mm(Top) + 1mm(Bottom)
  * **Power:**
      * I/O: 3.3V
      * Input: 5V
        

-----

## Documentation

  * [VSDSquadron PRO Datasheet](datasheet.pdf)
  * [SiFive FE310-G002 Datasheet v1p2](fe310-g002-datasheet-v1p2.pdf)
  * [SiFive FE310-G002 Manual v1p5](manual_fe310-g002-v1p5.pdf)
