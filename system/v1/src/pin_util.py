# pin_util.py  (simple functions, no classes)
# BOARD numbering (physical pins)

import time
import Jetson.GPIO as GPIO

FORBIDDEN_PINS = {
    1, 2, 4,                      # 3.3V / 5V
    6, 9, 14, 20, 25, 30, 34, 39,  # GND
    27, 28,                        # ID/I2C
}

def gpio_init():
    """Clean start + set BOARD mode once."""
    GPIO.cleanup()
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)

def setup_out(pin):
    """Setup one pin as output LOW."""
    if pin in FORBIDDEN_PINS:
        raise ValueError(f"Pin {pin} is forbidden (power/ground/ID)")
    GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

def high(pin):
    GPIO.output(pin, GPIO.HIGH)

def low(pin):
    GPIO.output(pin, GPIO.LOW)

def all_low(pins):
    for p in pins:
        low(p)

def all_high(pins):
    for p in pins:
        high(p)

def reset_array(pins, off_time=0.2, on_time=0.2):
    """All LOW -> wait -> all HIGH -> wait."""
    all_low(pins)
    time.sleep(off_time)
    all_high(pins)
    time.sleep(on_time)

def cleanup():
    GPIO.cleanup()
