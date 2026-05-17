import Jetson.GPIO as GPIO
import time

BUTTON_PIN = 29

def test_button():
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(BUTTON_PIN, GPIO.IN)

    print(f"Hardware Pull-up Active. Reading Pin {BUTTON_PIN}...")
    try:
        while True:
            val = GPIO.input(BUTTON_PIN)
            status = "PRESSED" if val == 0 else "RELEASED"
            print(f"Value: {val} | Status: {status}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    test_button()