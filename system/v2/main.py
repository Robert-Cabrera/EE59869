
import os
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"  

import sys
import logging
import onnxruntime as ort
from pathlib import Path
import time
import gc
import cv2

# Tests
sys.path.append(os.path.join(os.path.dirname(__file__), 'test'))
from test.test_frame import frame

# Srcs
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.debug_config import set_debug, debug_print, init_session
from src.frame_recorder import get_recorder
from src.process_env import process_env
from src.process_indoors import process_indoors
from src.process_outdoors import process_outdoors, init_outside_engine, get_outside_engine
from src.tof_thread import init_tof, get_distances, cleanup_tof
from src.camera import init_camera, get_frame, stop_camera, show_frame, stop_monitor
from src.button import init_button, check_button, cleanup_button
from src.pin_util import gpio_init
from src.audio_module import announce_direction, announce_stop, announce_initializing, announce_ready, announce_paused, announce_resume, announce_street_if_new, announce_tof_unavailable, reset_street_detection, connect_to_bluetooth, set_bluetooth_mode, set_audio_volume, set_environment, speak
from src.motor import init_motors, pulse_direction, cleanup_motors
from src.image_pp import draw_direction_arrow

# Global variables

# Debug flags default values
DEBUG   = False
USE_TOF = False  
MONITOR = False
USE_BLUETOOTH = True
RECORD = False
DEBUG_CAMERA = False

# Environment processing state
env_result = None
env_init = False

# Sensors
distances = {}
stop_signal = False
TOF_THRESHOLD = 200.00 # in mm
PRINT_THRESHOLD = 5

# Navigation
command = None
last_stop_announcement = 0  # Track timing for stop announcements
stop_active = False  # Track if obstacle is currently detected
print_cnt = 0

# Model loading control
LOAD_ADVANCED_MODELS = 0  # Set to 1 to load crosswalk/traffic/people models

# Environment mode control
ENV_MODE = "indoor"  # Use "auto" for automatic detection, or hardcode "indoor"/"outdoor"

# Frame skipping for efficient processing
INDOOR_OUTDOOR_FRAME_SKIP = 5  # Run navigation AI every 10 frames
navigation_frame_counter = 0
last_indoor_command = None
last_outdoor_command = None

ADVANCED_DETECTION_FRAME_SKIP = 20  # Run crosswalk/traffic detection every 20 frames
advanced_detection_counter = 0

# People detection announcement tracking
PEOPLE_ANNOUNCE_INTERVAL = 20  # Announce every 20 frames if person nearby
PEOPLE_ANNOUNCE_COOLDOWN = 15  # Don't mention again for 15 seconds
people_frame_counter = 0
last_people_announcement_time = 0

def load_models(model_dir=None, max_retries=3, environment=None):

    if model_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(script_dir, 'models')
    
    # Use provided environment or fall back to ENV_MODE
    env_mode = environment if environment is not None else ENV_MODE
    
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    models = {}
    model_path = Path(model_dir)
    
    # Find all .onnx files
    onnx_files = sorted(model_path.glob("*.onnx"))
    
    # Models to skip
    advanced_models = {'crosswalk_detect', 'traffic_light_world', 'people_detect', 'crosswalk_detector'}
    skip_models = set()
    
    # Skip clip_rn50 if env_mode is hardcoded (not auto)
    if env_mode != "auto":
        skip_models.add('clip_rn50')
    
    # Skip indoor/outdoor models based on environment
    if env_mode == "indoor":
        skip_models.add('fscnn_best')  # Skip outdoor model
        debug_print("[AI] Loading indoor-only models")
    elif env_mode == "outdoor":
        skip_models.add('miDas_indoors')  # Skip indoor model
        debug_print("[AI] Loading outdoor-only models")
    # If env_mode == "auto", load both models
    
    # Skip advanced models if disabled
    if not LOAD_ADVANCED_MODELS:
        skip_models.update(advanced_models)
    
    # Load each model sequentially with retry on memory errors
    for model_file in onnx_files:
        model_name = model_file.stem
        
        # Skip models
        if model_name in skip_models:
            debug_print(f"[AI] Skipping: {model_name}")
            continue
        
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
                debug_print(f"[AI] Loaded: {model_name} on {provider}")
                loaded = True
                break
                
            except RuntimeError as e:
                # Handle out-of-memory errors
                if "Failed to allocate memory" in str(e) or "ORT_OUT_OF_MEMORY" in str(e):
                    debug_print(f"[AI] Memory error loading {model_name} (attempt {attempt + 1}/{max_retries}): {e}")
                    gc.collect()
                    if attempt < max_retries - 1:
                        debug_print(f"[AI] Retrying in 2 seconds...")
                        time.sleep(2)
                    else:
                        debug_print(f"[AI] Failed to load {model_name} after {max_retries} attempts")
                else:
                    debug_print(f"[AI] Error loading {model_file}: {e}")
                    break
                    
            except Exception as e:
                debug_print(f"[AI] Error loading {model_file}: {e}")
                break

    traffic_models_dict = None
    if LOAD_ADVANCED_MODELS:
        traffic_models_dict = {
            'crosswalk_detect': models.get('crosswalk_detect'),
            'traffic_light_world': models.get('traffic_light_world'),
            'people_detect': models.get('people_detect')
        }
    
    return (models.get('clip_rn50'),
            models.get('miDas_indoors'),
            models.get('fscnn_best'),
            models.get('crosswalk_detector') if LOAD_ADVANCED_MODELS else None,
            traffic_models_dict)



if __name__ == "__main__":
    # Initialize session directory (creates logs/YYYY-MM-DD_HH-MM-SS/)
    # All debug_print calls will log to this session, regardless of flags
    init_session()
    
    # Parse command line arguments for debug flags
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == "--debug":
                DEBUG = True
                set_debug(True)
                debug_print("[Main] Debug mode enabled")
            elif arg == "--debug_camera":
                DEBUG_CAMRERA = True
                debug_print("[Main] Camera debug mode is enabled - hardcoded frames")
            elif arg == "--use-tof":
                USE_TOF = True
                debug_print("[Main] Using ToF sensors")
            elif arg == "--monitor":
                MONITOR = True
                debug_print("[Main] Video monitor enabled")
            elif arg == "--use-bluetooth":
                USE_BLUETOOTH = True
                debug_print("[Main] Bluetooth connection enabled")
            elif arg == "--record":
                RECORD = True
                debug_print("[Main] Frame recording enabled")
    
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
    
    # Init [AI] - returns environment/indoor/outdoor/crosswalk models + traffic models dict
    env_model, indoor_model, outdoor_model, crosswalk_model, traffic_models = load_models(environment=ENV_MODE)
    
    # Set audio module environment
    set_environment(ENV_MODE)
    
    # Print environment mode
    if ENV_MODE != "auto":
        debug_print(f"[Main] Environment mode: {ENV_MODE} (hardcoded)")
    else:
        debug_print(f"[Main] Environment mode: auto (detected)")
    
    # Init [Traffic Engine] with preloaded traffic/people models
    if traffic_models is not None:
        init_outside_engine(models_dict=traffic_models)
    
    # Init [Button] - with retry logic for GPIO conflicts/mode loss
    button_init_retries = 0
    while button_init_retries < 3:
        try:
            init_button()
            break
        except Exception as e:
            button_init_retries += 1
            error_msg = str(e)
            
            if "Device or resource busy" in error_msg or "Please set pin numbering mode" in error_msg:
                debug_print(f"[Main] GPIO error, retrying button init ({button_init_retries}/3)...")
                time.sleep(0.5)
                try:
                    # Cleanup and reinitialize GPIO mode
                    from src.pin_util import cleanup
                    cleanup()
                except:
                    pass
                gpio_init()
            else:
                raise
    
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
            debug_print(f"[TOF] Failed to initialize: {e}")
            announce_tof_unavailable()
            time.sleep(3)
            USE_TOF = False
            debug_print("[TOF] Disabling TOF sensors and continuing without them")

    # Init [Camera]
    if not DEBUG_CAMERA:
        init_camera("/dev/video0", w=640, h=480, fps=30, mjpg=True)
        time.sleep(1)
    
    # Announce ready to engage
    announce_ready()
    
    # Wait for button press to start
    debug_print("[Main] Waiting for button press to start...")
    button_state = False  # Track button state
    while not button_state:
        try:
            button_state = check_button()
        except RuntimeError as e:
            # GPIO mode or pin setup might have been lost, reset it
            if "Please set pin numbering mode" in str(e) or "must setup()" in str(e):
                gpio_init()
                init_button()
                debug_print("[Main] Reset GPIO mode and button")
            else:
                raise
        time.sleep(0.1)
    debug_print("[Main] Button pressed - starting operation")
    time.sleep(0.2)  # Debounce
    
    frame_idx = 0
    last_button_state = True  # Button starts pressed
    
    # Initialize frame recorder if enabled
    recorder = get_recorder()
    if RECORD:
        recorder.enable()
        recorder.start_session()

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
                debug_print("[Main] System paused - waiting for button press to resume")
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
                debug_print("[Main] Button pressed - resuming operation")
                announce_resume()
                reset_street_detection()  # Reset street detection state when resuming
                time.sleep(0.2)  # Debounce
                current_button_state = True
            
            last_button_state = current_button_state
                
            # Check sensors
            if USE_TOF:
                try:
                    distances = get_distances()
                    stop_signal = False  # Reset to False first
                    
                    # Check each sensor, but only if we have valid data (not None)
                    if distances['center'] is not None and distances['center'] < TOF_THRESHOLD:
                        stop_signal = True
                    if distances['left'] is not None and distances['left'] < TOF_THRESHOLD:
                        stop_signal = True
                    if distances['right'] is not None and distances['right'] < TOF_THRESHOLD:
                        # stop_signal = True
                        pass # Right sensor is defective 
                        pass
                except Exception as e:
                    debug_print(f"[TOF] Error reading sensors: {e}")
                    stop_signal = False
            else:
                stop_signal = False
            
            # Normal Operation ==============================================================
            if not stop_signal:
                
                # Get frame
                if DEBUG_CAMERA:
                    selFrame = frame(n=frame_idx % 5, indoors=False)
                else:
                    selFrame, frame_ts = get_frame(resize=(640, 480), rgb=False)
                    if selFrame is None:
                        debug_print(f"[AI] Frame {frame_idx}: No frame captured yet")
                        time.sleep(0.005)
                        continue

                # Process environment
                if ENV_MODE != "auto":
                    # Use hardcoded environment mode
                    if ENV_MODE in ["indoor", "outdoor"]:
                        env_result = ENV_MODE
                    else:
                        debug_print(f"[Main] Invalid ENV_MODE '{ENV_MODE}', use 'auto', 'indoor', or 'outdoor'")
                        env_result = "unknown"
                else:
                    # Auto-detect environment
                    env_result = process_env(env_model, selFrame)
                
                # Once environment is determined, continue with pipeline
                if env_result != "unknown":
                    env_init = True
                    # Update audio module with current environment
                    set_environment(env_result)
                    # Determine the appropiate pipeline based on environment
                    if env_result == "indoor" or env_result == "outdoor":
                        # Run indoor navigation AI every INDOOR_OUTDOOR_FRAME_SKIP frames
                        navigation_frame_counter += 1
                        if navigation_frame_counter % INDOOR_OUTDOOR_FRAME_SKIP == 0:
                            command = process_indoors(indoor_model, selFrame)
                            last_indoor_command = command
                        else:
                            command = last_indoor_command
                        
                        if frame_idx % PRINT_THRESHOLD == 0:
                            debug_print(f"[AI] Frame {frame_idx}: Indoor - Command: {command}")
                        
                        # Display frame with directional arrow if monitor is enabled
                        if MONITOR:
                            display_frame = draw_direction_arrow(selFrame, command, environment="indoor")
                            show_frame(display_frame, window="Video Monitor", delay=1)
                        else:
                            display_frame = draw_direction_arrow(selFrame, command, environment="indoor")
                        
                        # Save frame if recording is enabled
                        if recorder.is_enabled():
                            recorder.save_frame(display_frame, command, "Indoor")
                        
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
                        # Run outdoor navigation AI every INDOOR_OUTDOOR_FRAME_SKIP frames
                        navigation_frame_counter += 1
                        # Run advanced detection (crosswalk) every ADVANCED_DETECTION_FRAME_SKIP frames
                        advanced_detection_counter += 1
                        cw_model = crosswalk_model if advanced_detection_counter % ADVANCED_DETECTION_FRAME_SKIP == 0 else None
                        
                        if navigation_frame_counter % INDOOR_OUTDOOR_FRAME_SKIP == 0:
                            command, street_detected = process_outdoors(outdoor_model, selFrame, cw_model)
                            last_outdoor_command = command
                        else:
                            command = last_outdoor_command
                            # street_detected maintains its previous value via the buffer in process_outdoors
                            street_detected = False  # Will be updated only on detection frames
                        
                        if frame_idx % PRINT_THRESHOLD == 0:
                            debug_print(f"[AI] Frame {frame_idx}: Outdoor - Command: {command}, Street Detected: {street_detected}")
                        
                        # Display frame with directional arrow if monitor is enabled
                        if MONITOR:
                            display_frame = draw_direction_arrow(selFrame, command, environment="outdoor")
                            show_frame(display_frame, window="Video Monitor", delay=1)
                        else:
                            display_frame = draw_direction_arrow(selFrame, command, environment="outdoor")
                        
                        # Save frame if recording is enabled
                        if recorder.is_enabled():
                            recorder.save_frame(display_frame, command, f"Outdoor - Street: {street_detected}")
                        
                        announced, direction = announce_direction(command)
                        announce_street_if_new(street_detected)
                        
                        # People detection: announce every PEOPLE_ANNOUNCE_INTERVAL frames if person nearby
                        people_frame_counter += 1
                        if people_frame_counter % PEOPLE_ANNOUNCE_INTERVAL == 0:
                            try:
                                outside_engine = get_outside_engine()
                                if outside_engine:
                                    # Convert frame to RGB for people detection
                                    if len(selFrame.shape) == 2:  # Grayscale
                                        frame_rgb = cv2.cvtColor(selFrame, cv2.COLOR_GRAY2RGB)
                                    else:
                                        frame_rgb = cv2.cvtColor(selFrame, cv2.COLOR_BGR2RGB)
                                    
                                    people_boxes = outside_engine.detect_people(frame_rgb)
                                    
                                    # Check cooldown before announcing
                                    current_time = time.time()
                                    if len(people_boxes) > 0 and (current_time - last_people_announcement_time) >= PEOPLE_ANNOUNCE_COOLDOWN:
                                        speak("There is a person nearby.")
                                        last_people_announcement_time = current_time
                                        people_frame_counter = 0  # Reset counter after announcement
                            except Exception as e:
                                debug_print(f"[AI] Error detecting people: {e}")
                        
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
            
            # Sensor Override =============================================================================================================
            else:
                current_time = time.time()
                
                if not stop_active:
                    # FIRST detection → immediate + interrupt
                    debug_print(f"[ToF] Distances: center={distances['center']}, left={distances['left']}, right={distances['right']}")
                    announce_stop()  # uses interrupt=True
                    last_stop_announcement = current_time
                    stop_active = True
                
                elif current_time - last_stop_announcement >= 5.0:
                    # Subsequent reminders → normal (no interrupt)
                    debug_print(f"[ToF] Distances: center={distances['center']}, left={distances['left']}, right={distances['right']}")
                    speak("Obstacle still ahead, please wait", interrupt=False)
                    last_stop_announcement = current_time
            
            frame_idx += 1
            # Reset stop timer when no obstacle
            if not stop_signal:
                last_stop_announcement = 0
                stop_active = False

            # ============================================================================================================================

    except KeyboardInterrupt:
        debug_print("\n[AI] Shutting down...")
    finally:
        recorder.stop_session()
        cleanup_tof()
        debug_print("[ToF] Cleaned up resources.")
        stop_camera()
        stop_monitor(window="Video Monitor")
        debug_print("[Camera] Cleaned up resources.")
        cleanup_button()
        cleanup_motors()
        

