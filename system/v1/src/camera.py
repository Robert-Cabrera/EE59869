#!/usr/bin/env python3
import cv2
import time
import threading

class CameraRT:
    def __init__(self, dev="/dev/video0", w=640, h=480, fps=30, mjpg=True):
        self.dev = dev
        self.w = int(w)
        self.h = int(h)
        self.fps = int(fps)
        self.mjpg = bool(mjpg)

        self.cap = None
        self.running = False
        self.thread = None

        self.lock = threading.Lock()
        self.latest = None      
        self.latest_ts = None   

    def start(self):
        self.cap = cv2.VideoCapture(self.dev, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera: {self.dev}")

        if self.mjpg:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # flush a few frames
        for _ in range(5):
            self.cap.read()

        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue

            ts = time.time()
            with self.lock:
                self.latest = frame
                self.latest_ts = ts

    def get_frame(self, resize=None, rgb=False):
        with self.lock:
            if self.latest is None:
                return None, None
            frame = self.latest.copy()
            ts = self.latest_ts

        if resize is not None:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_LINEAR)

        if rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame, ts

_cam = None

def init_camera(dev="/dev/video0", w=640, h=480, fps=30, mjpg=True):
    global _cam
    _cam = CameraRT(dev=dev, w=w, h=h, fps=fps, mjpg=mjpg)
    _cam.start()
    return _cam

def get_frame(resize=None, rgb=False):
    if _cam is None:
        raise RuntimeError("Camera not initialized. Call init_camera() first.")
    return _cam.get_frame(resize=resize, rgb=rgb)

def show_frame(frame, window="camera", delay=1):
    """Display a single frame in a window (sequential, no threading)"""
    if frame is None:
        return
    
    cv2.imshow(window, frame)
    k = cv2.waitKey(delay) & 0xFF
    if k == ord('q') or k == 27:
        cv2.destroyWindow(window)
        return False  # Signal to stop
    return True  # Continue

def stop_camera():
    global _cam
    if _cam is not None:
        _cam.stop()
        _cam = None

def stop_monitor(window="camera"):
    cv2.destroyAllWindows()

if __name__ == "__main__":
    init_camera("/dev/video0", w=640, h=480, fps=30, mjpg=True)

    try:
        while True:
            frame, ts = get_frame(rgb=False)
            if frame is not None:
                cont = show_frame(frame, window="camera", delay=1)
                if not cont:
                    break
            else:
                print("No frame captured yet")
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        stop_camera()
