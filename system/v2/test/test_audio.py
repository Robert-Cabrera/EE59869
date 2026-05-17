import subprocess
import os

PIPER_EXEC = os.path.expanduser("~/piper_tts/piper/piper")
MODEL_PATH = os.path.expanduser("~/piper_tts/en_US-amy-medium.onnx")

def speak(text, speed=1):
    device_alias = "plughw:CARD=Audio,DEV=0"
    command = (
        f'echo "{text}" | {PIPER_EXEC} --model {MODEL_PATH} --output_raw | '
        f'sox -t raw -r 22050 -e signed-integer -b 16 -c 1 - -t wav - speed {speed} | '
        f'aplay -D {device_alias} -q'
    )
    
    try:
        subprocess.Popen(command, shell=True)
    except Exception as e:
        print(f"TTS Error: {e}")

if __name__ == "__main__":
    # If the USB adapter is playing double speed, 0.5 will fix it
    speak("...Please keep right", speed=1)