# EE59869 - Jetson Orin Nano Bringup

This repository serves to integrate multiple sensors with a Jetson Orin Nano development board, including camera and Time-of-Flight (ToF) distance sensors.

This will then transition into the creation of an AI pipeline using the camera and the sensors

## Components

### Camera Module (`camera_code/`)

A thread-safe real-time camera capture class with the following features:

- Configurable resolution (default 640x480) and frame rate (default 30 fps)
- Multi-threaded frame buffering for low-latency access
- Thread-safe frame retrieval with timestamps

### Time-of-Flight Sensors (`tof_code/`)

Multi-sensor distance measurement system with 3 VL53L1X ToF sensors:

- Automatic sensor address assignment via XSHUT pins
- Configurable retry logic for sensor initialization
- Per-sensor address mapping and identification

- **Left sensor**   (address 0x30) - GPIO PIN 32
- **Center sensor** (address 0x31) - GPIO PIN 31  
- **Right sensor**  (address 0x29) - GPIO PIN 33

- SDA: PIN 3
- SCL: PIN 5
- Connected to I2C1 peripheral
- VDD: 3.3V, VSS: GND

### GPIO Scripts (`scripts/`)

**File:** `enable_gpio_pins.py`

Initializes GPIO pins on Jetson Orin Nano using devmem for direct memory access:
- Enables pins: 7, 15, 29 (NON-WORKING), 31, 32, 33 as outputs
- Uses memory-mapped I/O via `busybox devmem`
- Requires `sudo` privileges

**File:** `enable-gpio.service`

Systemd service unit for automatic GPIO initialization on boot.

**File:** `jetson-remote.sh`

Establishes an SSH connection to the Jetson Orin Nano with local port forwarding:
- Connects to `seniordesign@seniordesign.local`
- Forwards local port 4002 to remote port 4000 (`-L 4002:localhost:4000`)
- Useful for remote development and NoMachine access
- Default credentials: `seniordesign` / `seniordesign`
