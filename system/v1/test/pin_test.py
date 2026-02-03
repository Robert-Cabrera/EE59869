#!/usr/bin/env python3
import time
import Jetson.GPIO as GPIO

PIN = 29
testingINPUT = True

if (not testingINPUT):
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    GPIO.setup(PIN, GPIO.OUT)

    print("HIGH")
    GPIO.output(PIN, GPIO.HIGH)
    time.sleep(2)

    print("LOW")
    GPIO.output(PIN, GPIO.LOW)
    time.sleep(2)

    print("BLINK")
    for _ in range(10):
        GPIO.output(PIN, GPIO.HIGH)
        time.sleep(0.25)
        GPIO.output(PIN, GPIO.LOW)
        time.sleep(0.25)

    GPIO.cleanup()
    print("DONE")

else:
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    GPIO.setup(PIN, GPIO.IN)

    print("Reading pin state. Press Ctrl+C to exit.")
    try:
        while True:
            pin_state = GPIO.input(PIN)
            print(f"Pin {PIN} state: {'HIGH' if pin_state else 'LOW'}")
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        GPIO.cleanup()