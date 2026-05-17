import Jetson.GPIO as GPIO
import time

# Pins based on your previous setup
MOTOR_L = 7
MOTOR_R = 15

def pulse(pin):
    """Creates a sharp 'bzz-bzz' double pulse."""
    for _ in range(2):
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(0.5) # Pulse duration
        GPIO.output(pin, GPIO.LOW)
        time.sleep(0.05)  # Short gap between pulses

def test_haptics():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup([MOTOR_L, MOTOR_R], GPIO.OUT, initial=GPIO.LOW)

    print("Running bzz-bzz sequence. Press Ctrl+C to stop.")
    try:
        while True:
            print("Left motor pulse...")
            pulse(MOTOR_L)
            
            time.sleep(1.0) # 1 second delay between sides

            print("Right motor pulse...")
            pulse(MOTOR_R)
            
            time.sleep(1.0) # 1 second delay before restarting loop
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Final safety kill
        GPIO.output(MOTOR_L, GPIO.LOW)
        GPIO.output(MOTOR_R, GPIO.LOW)
        GPIO.cleanup()
        print("Cleanup done.")

if __name__ == "__main__":
    test_haptics()