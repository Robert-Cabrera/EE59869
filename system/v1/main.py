
import os
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "3"  

import sys
import onnxruntime as ort
from pathlib import Path
import time

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

# Global variables

# Debug flags
DEBUG   = True
USE_TOF = False  
MONITOR = False

# Environment processing state
env_result = None
env_init = False

# Sensors
distances = {}
stop_signal = False

# Navigation
command = None

def load_models(model_dir=None):

    if model_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(script_dir, 'models')
    
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    models = {}
    model_path = Path(model_dir)
    
    # Find all .onnx files
    onnx_files = sorted(model_path.glob("*.onnx"))
    
    # Load each model sequentially
    for model_file in onnx_files:
        try:
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.log_severity_level = 3  # Suppress warnings and info logs
            session = ort.InferenceSession(str(model_file), sess_options=so, providers=providers)
            model_name = model_file.stem
            models[model_name] = session
            provider = session.get_providers()[0]
            print(f"[AI] Loaded: {model_name} on {provider}")
        except Exception as e:
            print(f"Error loading {model_file}: {e}")

    return (models.get('clip_rn50'),
            models.get('miDas_indoors'),
            models.get('fscnn_best'),
            models.get('crosswalk_detector'))

if __name__ == "__main__":
    
    # Init [AI]
    env_model, indoor_model, outdoor_model, crosswalk_model = load_models()
    
    # Init [TOF]
    if USE_TOF:
        init_tof()
        time.sleep(2)

    # Init [Camera]
    if not DEBUG:
        init_camera("/dev/video0", w=640, h=480, fps=30, mjpg=True)
        time.sleep(1)
    
    frame_idx = 0

    try:
        while True:
            # Check sensors
            if USE_TOF:
                distances = get_distances()
                if (distances['center'] < 0.9 or distances['left'] < 0.9  or distances['right'] < 0.9) and env_init:
                    stop_signal = True
                else:
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
                        print(f"[AI] Frame {frame_idx}: Indoor - Command: {command}")

                    elif env_result == "outdoor":
                        command, street_detected = process_outdoors(outdoor_model, selFrame, crosswalk_model)
                        print(f"[AI] Frame {frame_idx}: Outdoor - Command: {command}, Street Detected: {street_detected}")
            
            # Sensor Override ==============================================================
            else:
                print(f"[ToF] Frame {frame_idx}: STOP - STOP - STOP - STOP")
            
            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[AI] Shutting down...")
    finally:
        cleanup_tof()
        print("[ToF] Cleaned up resources.")
        stop_camera()
        stop_monitor(window="Video Monitor")
        print("[Camera] Cleaned up resources.")
        

