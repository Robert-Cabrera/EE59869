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

**File:** `enable_pins.sh/enable_pins.service`
Sets the pinmux for pins 7, 15, 29, 31, 32, and 33 (General Purpose Pins) as following on startup making use of the busybox api
7  - OUT
15 - OUT
29 - IN
31 - OUT
32 - OUT
33 - OUT

**File:** `reclaim-spi-gpio.dts`
Device tree overlay that reclaims SPI header pins as GPIO on the Jetson Orin Nano. This overlay disables the SPI1 and SPI3 peripherals by configuring their pads for GPIO input mode with tristate disabled.

**Compilation:** Compile using:
```bash
dtc -O dtb -o reclaim-spi-gpio.dtbo reclaim-spi-gpio.dts
```bash
dtc -O dtb -o reclaim-spi-gpio.dtbo reclaim-spi-gpio.dts
```

Copy the compiled `.dtbo` file to the boot directory:
```bash
sudo cp reclaim-spi-gpio.dtbo /boot/dtb/overlays/
```

Then apply it using the Jetson IO configurator:
```bash
sudo /opt/nvidia/jetson-io/jetson-io.py
```

Navigate to the "Configure Jetson 40-pin header" option and select the overlay to enable it.

After reboot, verify the overlay was applied:
```bash
cat /boot/extlinux/extlinux.conf
```

**Reclaimed Pins:**
- SPI1 header pins (19, 21, 23, 24, 26) and SPI3 header pins (13, 16, 18, 22, 37) are reclaimed as GPIO


**File:** `jetson-remote.sh`

Establishes an SSH connection to the Jetson Orin Nano with local port forwarding:
- Connects to `seniordesign@seniordesign.local`
- Forwards local port 4002 to remote port 4000 (`-L 4002:localhost:4000`)
- Useful for remote development and NoMachine access
- Default credentials: `seniordesign` / `seniordesign`
