#!/usr/bin/env python3
import subprocess
import os
from collections import deque
import time
import json
import signal
import threading

PIPER_EXEC = os.path.expanduser("~/piper_tts/piper/piper")
MODEL_PATH = os.path.expanduser("~/piper_tts/en_US-amy-medium.onnx")

# Bluetooth configuration
USE_BLUETOOTH = False

# Hysteresis state
decision_buffer = deque(maxlen=10)  # Keep last 10 decisions
last_announced = None  # Track last announced direction
decision_count = 0  # Count decisions for reassurance announcements
current_process = None  # Track current audio process
street_was_detected = False  # Track previous street detection state

# Audio gating lock - prevents concurrent audio playback
audio_lock = threading.Lock()

def set_bluetooth_mode(use_bluetooth):
    """Configure whether to use Bluetooth for audio output."""
    global USE_BLUETOOTH
    USE_BLUETOOTH = use_bluetooth

def set_audio_volume(volume_percent=40):
    """Set the USB audio output volume via PulseAudio.
    
    Args:
        volume_percent: Volume level as percentage (0-150)
    """
    try:
        # Set volume for USB Audio device
        subprocess.run(
            [
                'pactl', 'set-sink-volume',
                'alsa_output.usb-Generic_USB_Audio_20210726905926-00.analog-stereo',
                f'{volume_percent}%'
            ],
            capture_output=True,
            timeout=5
        )
        print(f"[Audio] Set USB audio volume to {volume_percent}%")
    except Exception as e:
        print(f"[Audio] Warning: Could not set volume: {e}")

def connect_to_bluetooth(max_retries=3):
    """Connect to Cassette headphones with retries."""
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                ['/usr/local/bin/connect-cassette.sh'], 
                capture_output=True, 
                text=True,
                timeout=10
            )
            data = json.loads(result.stdout.strip())
            
            if data['connected']:
                print(f"[Bluetooth] Connected to Cassette (attempt {attempt + 1})")
                return True
            else:
                print(f"[Bluetooth] Connection failed (attempt {attempt + 1}/{max_retries})")
        except Exception as e:
            print(f"[Bluetooth] Error on attempt {attempt + 1}: {e}")
        
        if attempt < max_retries - 1:
            time.sleep(0.2)
    
    print(f"[Bluetooth] Failed to connect after {max_retries} attempts")
    return False

def speak(text, speed=1):
    """Speak text, skipping if audio is already playing.
    Uses a lock to prevent concurrent audio playback.
    If another request comes in while speaking, it's skipped.
    """
    global current_process, audio_lock
    
    # Try to acquire the audio lock without blocking
    # If we can't acquire it, audio is already playing so we skip
    if not audio_lock.acquire(blocking=False):
        print(f"[Audio] Skipping (already playing): {text}")
        return
    
    try:
        # 1. Kill the entire process group if it exists
        if current_process is not None and current_process.poll() is None:
            try:
                # Send signal to the process group (negative PID)
                os.killpg(os.getpgid(current_process.pid), signal.SIGTERM)
                current_process.wait(timeout=0.5)
            except Exception as e:
                print(f"[Audio] Error terminating: {e}")

        # Use PulseAudio for audio output (let PulseAudio route to correct device)
        # This works for both USB audio and Bluetooth devices
        if USE_BLUETOOTH:
            device_alias = "pulse"
        else:
            device_alias = "plughw:CARD=Audio,DEV=0"
        
        command = (
            f'echo "{text}" | {PIPER_EXEC} --model {MODEL_PATH} --output_raw | '
            f'sox -t raw -r 22050 -e signed-integer -b 16 -c 1 - -t wav - speed {speed} | '
            f'aplay -D {device_alias} -q'
        )
        
        try:
            current_process = subprocess.Popen(
                command, 
                shell=True, 
                preexec_fn=os.setsid  # This creates a new session/group
            )
            # Wait for the process to complete before releasing the lock
            current_process.wait()
        except Exception as e:
            print(f"[Audio] Error with PulseAudio: {e}")
            print(f"[Audio] Retrying with default device...")
            # Fallback to default if pulse fails
            command = (
                f'echo "{text}" | {PIPER_EXEC} --model {MODEL_PATH} --output_raw | '
                f'sox -t raw -r 22050 -e signed-integer -b 16 -c 1 - -t wav - speed {speed} | '
                f'aplay -q'
            )
            try:
                current_process = subprocess.Popen(
                    command, 
                    shell=True, 
                    preexec_fn=os.setsid
                )
                # Wait for the process to complete before releasing the lock
                current_process.wait()
            except Exception as e2:
                print(f"[Audio] Fatal TTS Error: {e2}")
    finally:
        # Always release the lock when done
        audio_lock.release()

def get_direction_category(command):
    """Map command to general direction category."""
    if command is None:
        return None
    cmd_upper = str(command).upper()
    
    if "LEFT" in cmd_upper:
        return "LEFT"
    elif "RIGHT" in cmd_upper:
        return "RIGHT"
    elif "FORWARD" in cmd_upper or "STRAIGHT" in cmd_upper:
        return "FORWARD"
    return None

def should_announce(current_decision):
    """Determine if we should announce based on hysteresis logic."""
    global last_announced, decision_count
    
    if current_decision is None:
        return False, None
    
    # Increment decision counter and reset buffer every 150 decisions for reassurance
    decision_count += 1
    if decision_count >= 150:
        decision_count = 0
        last_announced = None  # Reset to allow reassurance announcement
    
    # Add to buffer
    decision_buffer.append(current_decision)
    
    current_category = get_direction_category(current_decision)
    
    if current_category is None:
        return False, None
    
    # If this is the first announcement, announce it
    if last_announced is None:
        last_announced = current_category
        return True, current_decision
    
    # If category hasn't changed, don't announce
    if current_category == last_announced:
        return False, None
    
    # Category changed - check if this is confirmed by buffer
    buffer_list = list(decision_buffer)
    if len(buffer_list) < 2:
        return False, None
    
    # Count occurrences of current category in buffer
    current_count = sum(1 for d in buffer_list if get_direction_category(d) == current_category)
    buffer_size = len(buffer_list)
    
    # For FORWARD, require 5+ confirmations before announcing
    if current_category == "FORWARD":
        if current_count >= 5:
            last_announced = current_category
            return True, current_decision
        return False, None
    
    # For LEFT/RIGHT, announce on first detection of change
    last_announced = current_category
    return True, current_decision

def get_announcement(command):
    """Generate appropriate announcement based on command and context."""
    global last_announced
    should_say, decision = should_announce(command)
    
    if not should_say or decision is None:
        return None
    
    cmd_upper = str(decision).upper()
    category = get_direction_category(decision)
    
    # Check if this is a direction change
    is_direction_change = (last_announced is not None and 
                          last_announced != category and 
                          category in ["LEFT", "RIGHT"])
    
    if category == "FORWARD":
        return "It is all clear, keep forward"
    elif category == "LEFT":
        if is_direction_change:
            return "Please head left"
        else:
            return "Please keep left"
    elif category == "RIGHT":
        if is_direction_change:
            return " Please head right"
        else:
            return " Please keep right"
    
    return None

def announce_direction(command):
    """Check if we should announce and speak the direction.
    Allows interruption of previous speech.
    """
    global current_process
    
    announcement = get_announcement(command)
    if announcement:
        direction = get_direction_category(command)
        print(f"[Audio] Announcing: {announcement} ({direction})")
        
        speak(announcement)
        return True, direction
    return False, None

def announce_stop():
    """Announce stop command immediately, interrupting anything else."""
    print("[Audio] Announcing: STOP")
    
    # Emergency stops should always interrupt immediately
    speak("Obstacle very near, please stop")
    return True

def announce_initializing():
    """Announce system is initializing."""
    print("[Audio] Announcing: Initializing")
    speak("Please wait...")
    time.sleep(0.5)  
    speak("Initializing the system...")

def announce_ready():
    """Announce system is ready to engage."""
    print("[Audio] Announcing: Ready to engage")
    speak("Thank you for waiting. The system is ready!")

def announce_paused():
    """Announce system is paused."""
    print("[Audio] Announcing: Paused")
    speak("The system has been paused. Please press the button to resume.")

def announce_resume():
    """Announce system is resuming."""
    print("[Audio] Announcing: Resuming")
    speak("Resuming operation. Please be cautious.")

def announce_tof_unavailable():
    """Announce that TOF sensors are unavailable."""
    print("[Audio] Announcing: TOF Unavailable")
    speak("Proximity sensors not available. Please be extra careful.")

def announce_street_if_new(street_detected):
    """Announce street detected only on transition from not-detected to detected."""
    global street_was_detected
    
    # Only announce on transition from not-detected to detected
    if street_detected and not street_was_detected:
        print("[Audio] Announcing: Street Detected")
        speak("Warning: Street detected, please be cautious.")
    
    street_was_detected = street_detected

def reset_street_detection():
    """Reset street detection state (call when starting new navigation session)."""
    global street_was_detected
    street_was_detected = False