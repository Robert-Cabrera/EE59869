"""Pause button module for pausing the program."""

from pin_util import setup_in, read
from debug_config import debug_print
import Jetson.GPIO as GPIO

BUTTON_PIN = 29

def init_button():
    """Initialize GPIO for pause button on pin 29."""
    setup_in(BUTTON_PIN, pull=GPIO.PUD_UP)
    debug_print(f"[Button] Initialized pause button on pin {BUTTON_PIN}")

def check_button():
    """Check if pause button is pressed (returns True if pressed)."""
    val = read(BUTTON_PIN)
    return val == 0  # Button pressed when pin is LOW

def cleanup_button():
    """Clean up GPIO resources."""
    GPIO.cleanup()
    debug_print("[Button] Cleaned up GPIO resources.")
