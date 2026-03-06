
import os
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"  

import sys
import onnxruntime as ort
from pathlib import Path
import time
import gc

# Tests
sys.path.append(os.path.join(os.path.dirname(__file__), 'test'))
from test.test_frame import frame

# Srcs
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.process_env import process_env
from src.process_indoors import process_indoors
from src.process_outdoors import process_outdoors
from src.tof_thread import init_tof, get_distances, cleanup_tof
from src.camera import init_camera, get_frame, stop_camera, show_frame, stop_monitor
from src.button import init_button, check_button, cleanup_button
from src.pin_util import gpio_init
from src.audio_module import announce_direction, announce_stop, announce_initializing, announce_ready, announce_paused, announce_resume, announce_street_if_new, announce_tof_unavailable, reset_street_detection, connect_to_bluetooth, set_bluetooth_mode, set_audio_volume
from src.motor import init_motors, pulse_direction, cleanup_motors

# Global variables

# Debug flags default values
DEBUG   = False
DEBUG_PRINT = True
USE_TOF = False  
MONITOR = False
USE_BLUETOOTH = True

# Environment processing state
env_result = None
env_init = False

# Sensors
distances = {}
stop_signal = False
TOF_THRESHOLD = 600.00 # in mm

# Navigation
command = None
last_stop_announcement = 0  # Track timing for stop announcements

def load_models(model_dir=None, max_retries=3):

    if model_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(script_dir, 'models')
    
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    models = {}
    model_path = Path(model_dir)
    
    # Find all .onnx files
    onnx_files = sorted(model_path.glob("*.onnx"))
    
    # Load each model sequentially with retry on memory errors
    for model_file in onnx_files:
        model_name = model_file.stem
        loaded = False
        
        for attempt in range(max_retries):
            try:
                # Force garbage collection before loading
                gc.collect()
                
                so = ort.SessionOptions()
                so.intra_op_num_threads = 1
                so.log_severity_level = 3  # Suppress warnings and info logs
                session = ort.InferenceSession(str(model_file), sess_options=so, providers=providers)
                models[model_name] = session
                provider = session.get_providers()[0]
                print(f"[AI] Loaded: {model_name} on {provider}")
                loaded = True
                break
                
            except RuntimeError as e:
                # Handle out-of-memory errors
                if "Failed to allocate memory" in str(e) or "ORT_OUT_OF_MEMORY" in str(e):
                    print(f"[AI] Memory error loading {model_name} (attempt {attempt + 1}/{max_retries}): {e}")
                    gc.collect()
                    if attempt < max_retries - 1:
                        print(f"[AI] Retrying in 2 seconds...")
                        time.sleep(2)
                    else:
                        print(f"[AI] Failed to load {model_name} after {max_retries} attempts")
                else:
                    print(f"[AI] Error loading {model_file}: {e}")
                    break
                    
            except Exception as e:
                print(f"[AI] Error loading {model_file}: {e}")
                break

    return (models.get('clip_rn50'),
            models.get('miDas_indoors'),
            models.get('fscnn_best'),
            models.get('crosswalk_detector'))

if __name__ == "__main__":
    # Parse command line arguments for debug flags
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == "--debug":
                DEBUG = True
                print("[Main] Debug mode enabled")
            elif arg == "--debug-print":
                DEBUG_PRINT = True
                print("[Main] Debug print enabled")
            elif arg == "--use-tof":
                USE_TOF = True
                print("[Main] Using ToF sensors")
            elif arg == "--monitor":
                MONITOR = True
                print("[Main] Video monitor enabled")
            elif arg == "--use-bluetooth":
                USE_BLUETOOTH = True
                print("[Main] Bluetooth connection enabled")
    
    # Configure audio module for Bluetooth if enabled
    set_bluetooth_mode(USE_BLUETOOTH)
    
    # Set audio volume early 
    set_audio_volume(40)
    
    # Init [Bluetooth] startup announcements go through the right device
    if USE_BLUETOOTH:
        connect_to_bluetooth()
    
    # Init [GPIO] - must be called first
    gpio_init()
    
    # Announce initialization starting
    announce_initializing()
    
    # Init [AI]
    env_model, indoor_model, outdoor_model, crosswalk_model = load_models()
    
    # Init [Button]
    init_button()
    
    # Init [Motors]
    init_motors()
    
    # Init [TOF]
    if USE_TOF:
        try:
            from src.tof_thread import get_tof_thread
            init_tof()
            # Wait longer for initialization to complete
            time.sleep(3)
            # Check if initialization failed in the thread
            tof_thread = get_tof_thread()
            if tof_thread and tof_thread.initialization_error:
                raise Exception(tof_thread.initialization_error)
            if tof_thread and not tof_thread.running:
                raise Exception("ToF thread failed to start")
        except Exception as e:
            print(f"[TOF] Failed to initialize: {e}")
            announce_tof_unavailable()
            time.sleep(3)
            USE_TOF = False
            print("[TOF] Disabling TOF sensors and continuing without them")

    # Init [Camera]
    if not DEBUG:
        init_camera("/dev/video0", w=640, h=480, fps=30, mjpg=True)
        time.sleep(1)
    
    # Announce ready to engage
    announce_ready()
    
    # Wait for button press to start
    print("[Main] Waiting for button press to start...")
    button_state = False  # Track button state
    while not button_state:
        try:
            button_state = check_button()
        except RuntimeError as e:
            # GPIO mode or pin setup might have been lost, reset it
            if "Please set pin numbering mode" in str(e) or "must setup()" in str(e):
                gpio_init()
                init_button()
                print("[Main] Reset GPIO mode and button")
            else:
                raise
        time.sleep(0.1)
    print("[Main] Button pressed - starting operation")
    time.sleep(0.2)  # Debounce
    
    frame_idx = 0
    last_button_state = True  # Button starts pressed

    try:
        while True:

            # Check button state for transitions
            try:
                current_button_state = check_button()
            except RuntimeError as e:
                if "Please set pin numbering mode" in str(e) or "must setup()" in str(e):
                    gpio_init()
                    init_button()
                    init_motors()
                    current_button_state = check_button()
                else:
                    raise
            
            # Detect button release (transition from pressed to released)
            if last_button_state and not current_button_state:
                # Button released, pause the entire loop
                announce_paused()
                reset_street_detection()  # Reset street detection state when pausing
                print("[Main] System paused - waiting for button press to resume")
                # Wait for button press to resume
                while True:
                    try:
                        if check_button():
                            break
                    except RuntimeError as e:
                        if "Please set pin numbering mode" in str(e) or "must setup()" in str(e):
                            gpio_init()
                            init_button()
                            init_motors()
                        else:
                            raise
                    time.sleep(0.1)
                print("[Main] Button pressed - resuming operation")
                announce_resume()
                reset_street_detection()  # Reset street detection state when resuming
                time.sleep(0.2)  # Debounce
                current_button_state = True
            
            last_button_state = current_button_state
                
            # Check sensors
            if USE_TOF:
                try:
                    distances = get_distances()
                    if DEBUG_PRINT:
                        print(f"[ToF] Distances: center={distances['center']}, left={distances['left']}, right={distances['right']}")
                    
                    stop_signal = False  # Reset to False first
                    
                    # Check each sensor, but only if we have valid data (not None)
                    if distances['center'] is not None and distances['center'] < TOF_THRESHOLD:
                        stop_signal = True
                    if distances['left'] is not None and distances['left'] < TOF_THRESHOLD:
                        stop_signal = True
                    if distances['right'] is not None and distances['right'] < TOF_THRESHOLD:
                        stop_signal = True
                except Exception as e:
                    print(f"[TOF] Error reading sensors: {e}")
                    stop_signal = False
            else:
                stop_signal = False
            
            # Normal Operation ==============================================================
            if not stop_signal:
                
                # Get frame
                if DEBUG:
                    selFrame = frame(n=frame_idx % 5, indoors=False)
                else:
                    selFrame, frame_ts = get_frame(resize=(640, 480), rgb=False)
                    if selFrame is None:
                        print(f"[AI] Frame {frame_idx}: No frame captured yet")
                        time.sleep(0.005)
                        continue

                if MONITOR:
                    show_frame(selFrame, window="Video Monitor", delay=1)

                # Process environment
                env_result = process_env(env_model, selFrame)
                
                # Once environment is determined, continue with pipeline
                if env_result != "unknown":
                    env_init = True
                    # Determine the appropiate pipeline based on environment
                    if env_result == "indoor":
                        command = process_indoors(indoor_model, selFrame)
                        if DEBUG_PRINT:
                            print(f"[AI] Frame {frame_idx}: Indoor - Command: {command}")
                        announced, direction = announce_direction(command)
                        if announced and direction:
                            try:
                                # pulse_direction(direction)
                                pass
                            except RuntimeError as e:
                                if "not been set up" in str(e):
                                    gpio_init()
                                    init_button()
                                    init_motors()
                                    # pulse_direction(direction)
                                else:
                                    raise

                    elif env_result == "outdoor":
                        command, street_detected = process_outdoors(outdoor_model, selFrame, crosswalk_model)
                        
                        if DEBUG_PRINT:
                            print(f"[AI] Frame {frame_idx}: Outdoor - Command: {command}, Street Detected: {street_detected}")
                        
                        announced, direction = announce_direction(command)
                        announce_street_if_new(street_detected)
                        
                        if announced and direction:
                            try:
                                #pulse_direction(direction)
                                pass
                            except RuntimeError as e:
                                if "not been set up" in str(e):
                                    gpio_init()
                                    init_button()
                                    init_motors()
                                    # pulse_direction(direction)
                                else:
                                    raise
            
            # Sensor Override ==============================================================
            else:
                print(f"[ToF] Frame {frame_idx}: STOP - STOP - STOP - STOP")
                # Announce stop only when first entering the stop state
                if not last_stop_announcement:
                    announce_stop()
                    last_stop_announcement = time.time()
            
            frame_idx += 1
            # Reset stop timer when no obstacle
            if not stop_signal:
                last_stop_announcement = 0

    except KeyboardInterrupt:
        print("\n[AI] Shutting down...")
    finally:
        cleanup_tof()
        print("[ToF] Cleaned up resources.")
        stop_camera()
        stop_monitor(window="Video Monitor")
        print("[Camera] Cleaned up resources.")
        cleanup_button()
        cleanup_motors()
        

