"""Motor control module for haptic feedback."""

import Jetson.GPIO as GPIO
import time

MOTOR_L = 7
MOTOR_R = 15
motors_initialized = False

def init_motors():
    """Initialize GPIO for motors."""
    global motors_initialized
    GPIO.setup([MOTOR_L, MOTOR_R], GPIO.OUT, initial=GPIO.LOW)
    motors_initialized = True
    print("[Motor] Initialized motors on pins L={} R={}".format(MOTOR_L, MOTOR_R))

def pulse(pin, duration=0.2, gap=0.05):
    """Creates a double pulse on a motor pin."""
    for _ in range(2):
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(duration)
        GPIO.output(pin, GPIO.LOW)
        time.sleep(gap)
    # Actively pull down to ensure motor is off
    GPIO.output(pin, GPIO.LOW)
    time.sleep(0.01)  # Brief settling time

def pulse_direction(direction):
    """Pulse motor(s) based on direction.
    
    Args:
        direction: str - "LEFT", "RIGHT", or "FORWARD"
    """
    if not motors_initialized:
        return
    
    if direction == "LEFT":
        print("[Motor] Pulsing LEFT motor")
        pulse(MOTOR_L)
    elif direction == "RIGHT":
        print("[Motor] Pulsing RIGHT motor")
        pulse(MOTOR_R)
    elif direction == "FORWARD":
        # No need for attention
        pass

def cleanup_motors():
    """Stop all motors and clean up GPIO."""
    global motors_initialized
    if motors_initialized:
        try:
            # Make sure GPIO mode is set before trying to output
            try:
                GPIO.setmode(GPIO.BOARD)
            except:
                pass  # Mode might already be set
            
            # Try to setup pins again in case they weren't initialized
            try:
                GPIO.setup([MOTOR_L, MOTOR_R], GPIO.OUT, initial=GPIO.LOW)
            except:
                pass  # Pins might already be set up
            
            # Now set them to LOW
            try:
                GPIO.output(MOTOR_L, GPIO.LOW)
                GPIO.output(MOTOR_R, GPIO.LOW)
            except:
                pass  # Pins might not be accessible
                
            motors_initialized = False
            print("[Motor] Cleaned up motors")
        except Exception as e:
            motors_initialized = False
            print(f"[Motor] Error during cleanup: {e}")
