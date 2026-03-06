import sys
import os

# Add the 'tof' subdirectory to the path so we can find the .so file
current_dir = os.path.dirname(os.path.abspath(__file__))
tof_dir = os.path.join(current_dir, 'tof')
if tof_dir not in sys.path:
    sys.path.append(tof_dir)

try:
    import tof_driver
except ImportError as e:
    print(f"[ToF] Critical Error: Could not find tof_driver.so in {tof_dir}")
    raise e

class MockTofThread:
    def __init__(self):
        self.initialization_error = None
        self.running = True

_thread_handle = MockTofThread()

def init_tof():
    print("[ToF] Initializing C++ Driver...")
    try:
        tof_driver.init_tof()
        print("[ToF] Success.")
    except Exception as e:
        _thread_handle.initialization_error = str(e)
        _thread_handle.running = False
        print(f"[ToF] Init Failed: {e}")

def get_distances():
    # C++ returns {'right': val, 'center': val, 'left': val}
    try:
        return tof_driver.get_distances()
    except Exception as e:
        print(f"[ToF] Read Error: {e}")
        return {'right': 9999, 'center': 9999, 'left': 9999}

def cleanup_tof():
    tof_driver.cleanup_tof()

def get_tof_thread():
    return _thread_handle