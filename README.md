# VSDSquadron PRO Board Documentation

This document outlines the hardware setup, software installation, basic program validation, and key specifications for the VSDSquadron PRO Board.

-----

## I. Hardware Setup & Software Installation

### RISC-V Toolchain and Required Software

Install the necessary drivers and development environment for the VSDSquadron PRO board.

#### Zadig Driver Installation

1.  **Download Zadig:** Obtain the `zadig-2.5.exe` utility.
2.  **Connect Board:** Plug the VSDSquadron PRO board into the PC.
3.  **Launch Zadig:** Run `zadig-2.5.exe`.
4.  **Select Device:** Go to `Options > List All Devices` and select `Future Technology Devices International, Ltd.`.
5.  **Install Driver:** Choose `libusb-win32 (v1.2.6.0)` from the driver list and click `Replace Driver`. Confirm the installation.
6.  **Verify:** After successful installation, the device should appear under `Universal Serial Bus devices` in Device Manager as `VSD Squadron PRO`.

#### Freedom Studio Setup

1.  **Download Freedom Studio:** Download the `Freedom Studio Tar.gz` file for VSDSquadron PRO.
2.  **Extract:** Extract the contents of the `Tar.gz` file to a preferred location.
3.  **Launch Freedom Studio:**
      * **Windows:** Navigate to the extracted `FreedomStudio` folder and run `FreedomStudio.exe`.
      * **Linux:** Open a terminal in the extracted `FreedomStudio` directory and execute `./FreedomStudio`.

-----

## II. Basic Program Upload & Validation

### Testing 'sifive-welcome' Program

This section details the procedure for uploading and validating the `sifive-welcome` example program.

1.  **Open Freedom Studio:** Launch the Freedom Studio IDE.
2.  **Import Project:**
      * Go to `File > Import`.
      * Select `Existing Projects into Workspace` and click `Next`.
      * Browse to the `FreedomStudio` installation directory, then `examples > welcome_sifive`.
      * Ensure `Copy projects into workspace` is checked and click `Finish`.
3.  **Build Project:**
      * Right-click the `welcome_sifive` project in the `Project Explorer`.
      * Select `Build Project`.
      * Verify a successful build in the `Console` window.
4.  **Connect Board:** Ensure the VSDSquadron PRO board is connected to the PC via USB.
5.  **Program Board:**
      * Right-click the `welcome_sifive` project.
      * Select `Debug As > Freedom Studio J-Link Debugging`.
6.  **Program Execution:** The program will upload and execute.
7.  **Verify Output:**
      * Observe the `VSD Squadron PRO` serial port output in a terminal (e.g., PuTTY, Tera Term, or Freedom Studio's internal console).
      * Expected output: The program should display "Hello, World\!" and "Welcome to SiFive" messages.
      * **Note:** During device enumeration, you may encounter a prompt: "The connected J-Link is an EDU version. It is not licensed for commercial use. If you are using it for a commercial purpose, please contact SEGGER or its distributors." Select `OK` to continue.

-----

## III. Board Specifications

The VSDSquadron PRO board is powered by the SiFive FE310-G002 RISC-V SoC.

### FE310-G002 SiFive RISC-V SoC

  * **Core:** SiFive E31 (RV32IMAC)
      * **Architecture:** 32-bit RISC-V
      * **Frequency:** Up to 320 MHz
      * **Cache:** 16KB L1 Instruction Cache
      * **Pipelining:** 4-stage pipeline
      * **Multiplier/Divider:** Single-cycle hardware multiplier, hardware divider
  * **Debug:** JTAG Debug Interface (2-bit opcode, 7-bit debug module address, 32-bit data field)
      * IDCODE: 0x20000913

### Memory & Storage

  * **Data SRAM:** 16KB
  * **Flash Memory:** 128Mbit (16MB) QSPI Flash

### Peripherals & I/O

  * **USB:** USB to Serial converter (CP2102N)
  * **Clock:** On-board 16 MHz Crystal Oscillator
  * **LEDs:**
      * 3 User LEDs
      * 1 Debug LED
      * 1 Power LED
  * **Buttons:**
      * 1 Reset Button
      * 1 IO Button
  * **I/O Headers:** Through-hole headers for GPIO access.
  * **Antenna:** On-board Antenna
  * **Power:** 5V DC barrel jack and USB powered.
  * **Form Factor:** Arduino UNO R3 compatible.

-----

## Documentation

  * [VSDSquadron PRO Datasheet]()
  * [SiFive FE310-G002 Datasheet v1p2]()
  * [SiFive FE310-G002 Manual v1p5]()
