#!/usr/bin/env python3
import subprocess
import sys

# Pin to memory address mapping for Jetson Orin Nano
# Only the pins needed for this project
pin_memory_map = {
    7:  ("0x2448030", "0xA"),
    15: ("0x2440020", "0x5"),
    29: ("0x2430068", "0x8"),
    31: ("0x2430070", "0x8"),
    32: ("0x2434080", "0x5"),
    33: ("0x2434040", "0x4"),
}

print("Enabling GPIO pins as outputs using devmem...\n")

for pin, (addr, value) in pin_memory_map.items():
    try:
        # Use devmem to set pin to output mode
        subprocess.run(f"sudo busybox devmem {addr} w {value}", shell=True, check=True, capture_output=True)
        print(f"✓ Pin {pin}: Enabled as output")
    except Exception as e:
        print(f"Pin {pin}: Failed to enable - {str(e)}")

print("\n" + "="*50)
print("GPIO pins 7, 15, 29, 31, 32, 33 enabled!")
print("="*50)

