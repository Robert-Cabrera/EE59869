import cv2
import numpy as np
import torch
import time
from pathlib import Path
from collections import Counter
from src.image_pp import preprocess
from src.debug_config import debug_print
from src.audio_module import speak

# YOLO input dimensions
YOLO_INPUT_W = 512
YOLO_INPUT_H = 512

# Global state for ray casting
_ray_config = None
_ray_caster = None
_corridor_finder = None
_steering_cmd = None
_θ_cmd_smooth = 0.0

# Global state for crosswalk detection voting
_crosswalk_buffer = []
_crosswalk_buffer_size = 3  # Number of frames to keep for majority voting

# Global OutsideEngine instance
_outside_engine = None


class RayConfig:
    """Configuration for ray casting parameters."""
    def __init__(self, n_rays=90, fov_deg=40.0, max_range_frac=5, 
                 vertical_scale=0.11, min_obstacle_area=1000):
        self.n_rays = n_rays
        self.fov_deg = fov_deg
        self.max_range_frac = max_range_frac
        self.vertical_scale = vertical_scale
        self.min_obstacle_area = min_obstacle_area
        
    @property
    def angles(self):
        """Compute ray angles, corrected for vertical scale perspective distortion."""
        fov_rad = np.deg2rad(self.fov_deg / 2)
        fov_corrected = np.arctan(np.tan(fov_rad) * self.vertical_scale)
        fov_deg_corrected = np.rad2deg(fov_corrected) * 2
        return np.linspace(-fov_deg_corrected/2, fov_deg_corrected/2, self.n_rays)

    def get_max_range(self, image_height):
        """Compute max ray range based on image height."""
        return int(image_height * self.max_range_frac)


class RayCaster:
    """Performs ray casting collision detection from a given origin."""
    
    def __init__(self, config):
        self.config = config
        
    def cast_rays(self, mask, origin):
        h, w = mask.shape
        ox, oy = origin

        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        # --- remove small obstacle blobs (no contours) ---
        obstacles = (mask == 0).astype(np.uint8)  # 1 where obstacle
        num, labels, stats, _ = cv2.connectedComponentsWithStats(obstacles, connectivity=8)

        filtered_mask = mask.copy()
        min_area = int(self.config.min_obstacle_area)

        # (loop over components is fine; num is usually small)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                filtered_mask[labels == i] = 255  # erase small obstacle -> walkable

        # --- vectorized ray march (same aspect as original) ---
        angles = np.deg2rad(self.config.angles).astype(np.float32)
        vs = float(self.config.vertical_scale)

        # IMPORTANT: must account for slow y-motion when vs < 1
        # Need up to ~h/vs steps to reach y=0, plus some margin
        max_r = int(np.sqrt(w * w + (h / max(vs, 1e-6)) * (h / max(vs, 1e-6)))) + 2
        r = np.arange(max_r, dtype=np.float32)

        distances = []
        for θ in angles:
            dx = np.sin(θ)
            dy = -np.cos(θ)

            xs = (ox + r * dx).astype(np.int32)
            ys = (oy + r * dy * vs).astype(np.int32)  # <-- SAME as your original

            # In-bounds is a prefix (then all false), so trimming keeps correct "r index"
            inb = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            xs = xs[inb]
            ys = ys[inb]

            if xs.size == 0:
                distances.append(0)
                continue

            line = filtered_mask[ys, xs]
            hit = np.flatnonzero(line == 0)  # first obstacle along ray
            distances.append(int(hit[0]) if hit.size else int(xs.size))

        return distances


class CorridorFinder:
    """Finds navigable corridors using sliding window convolution."""
    
    def __init__(self, win_size=11, center_bias=0.7, prior_weight=0.15):
        self.win_size = win_size
        self.center_bias = center_bias
        self.prior_weight = prior_weight
        
    def sliding_window(self, distances, angles, weights=None):
        v = np.asarray(distances, np.float32)
        d = np.maximum(v, 0)
        d_norm = d / (d.max() + 1e-6)
        
        ang = np.asarray(angles, np.float32)
        n = len(d_norm)
        half = self.win_size // 2
        
        if weights is None:
            w = np.ones(self.win_size, np.float32) / self.win_size
        else:
            w = np.asarray(weights, np.float32)
            w /= (w.sum() + 1e-6)
        
        scores = np.empty(n, np.float32)
        windows = []
        
        for i_center in range(n):
            idxs = [(i_center + j - half) % n for j in range(self.win_size)]
            windows.append(idxs)
            scores[i_center] = float(d_norm[idxs] @ w)
        
        # Apply forward bias (cosine prior)
        prior = np.cos(np.deg2rad(ang))
        prior = (prior - prior.min()) / (prior.max() - prior.min() + 1e-6)
        scores = scores + self.prior_weight * prior
        
        i_best = int(np.argmax(scores))
        best_corridor = windows[i_best]
        θ_best = float(ang[i_best])
        
        if n >= 2:
            part = np.partition(scores, -2)
            confidence = (float(part[-1]) - float(part[-2])) / (float(part[-1]) + 1e-6)
        else:
            confidence = 1.0
        confidence = max(confidence, 0.05)
        
        return scores, best_corridor, θ_best, confidence


class SteeringCommand:
    """Converts steering angles into discrete movement commands."""
    def __init__(self, frame_buffer_size=20, inertia_weight=20):
        self.frame_buffer_size = frame_buffer_size
        self.inertia_weight = inertia_weight
        self.last_cmd = "FORWARD"
        self.command_buffer = []
        self.history = []
        self.current_zone = "FORWARD"

    def angle_to_cmd(self, θ):
        """Hysteresis-based angle to command conversion."""
        if θ < -1.1: return "LEFT"
        if θ >  1.1: return "RIGHT"

        if self.current_zone == "SLIGHT_LEFT":
            if θ > -0.2: return "FORWARD"
            return "SLIGHT_LEFT"

        if self.current_zone == "SLIGHT_RIGHT":
            if θ < 0.2: return "FORWARD"
            return "SLIGHT_RIGHT"

        if self.current_zone == "FORWARD":
            if θ < -0.3: return "SLIGHT_LEFT"
            if θ > 0.3: return "SLIGHT_RIGHT"
            return "FORWARD"

        return "FORWARD"

    def is_opposite(self, a, b):
        left  = {"LEFT", "SLIGHT_LEFT"}
        right = {"RIGHT", "SLIGHT_RIGHT"}
        return (a in left and b in right) or (a in right and b in left)

    def compute(self, θ_cmd_smooth):
        """Compute steering command with inertia and buffering."""
        frame_cmd = self.angle_to_cmd(θ_cmd_smooth)
        self.current_zone = frame_cmd

        self.command_buffer.append(frame_cmd)
        if len(self.command_buffer) > self.frame_buffer_size:
            self.command_buffer.pop(0)

        vote_weights = Counter(self.command_buffer)

        if self.is_opposite(self.last_cmd, frame_cmd):
            vote_weights[self.last_cmd] += self.inertia_weight

        cmd = max(vote_weights, key=vote_weights.get)
        self.last_cmd = cmd
        self.history.append(cmd)
        if len(self.history) > 30:
            self.history.pop(0)

        return cmd


class CrosswalkTrafficEngine:
    """Manages crosswalk state and traffic light waiting with timeout."""
    
    def __init__(self, crosswalk_model=None, traffic_model=None):
        self.crosswalk_model = crosswalk_model
        self.traffic_model = traffic_model
        
        # Crosswalk state with hysteresis
        self.crosswalk_state = False
        self.crosswalk_counter = 0
        self.crosswalk_hysteresis_on = 2
        self.crosswalk_hysteresis_off = 3
        self.last_crosswalk_boxes = []
        
        # Traffic state with hysteresis
        self.traffic_state = "UNKNOWN"
        self.traffic_go_counter = 0
        self.traffic_hysteresis_go = 3
        
        # Crossing mode: lock system for 10 seconds after GO
        self.crossing_start_time = None
        self.crossing_hold_time = 10  # seconds
        
        # Traffic light timeout: allow crossing if stuck waiting > 30 seconds
        self.wait_start_time = None
        self.wait_timeout = 30  # seconds
        
        # Audio announcement tracking (prevent spamming)
        self.last_crosswalk_announced = False
        self.last_waiting_announced = False
        self.last_go_announced = False
        self.last_timeout_announced = False

    def reset(self):
        """Reset to initial scanning state."""
        self.crosswalk_state = False
        self.crosswalk_counter = 0
        self.last_crosswalk_boxes = []
        self.traffic_state = "UNKNOWN"
        self.traffic_go_counter = 0
        self.wait_start_time = None
        self.crossing_start_time = None
        # Reset audio flags so announcements can fire again on next cycle
        self.last_crosswalk_announced = False
        self.last_waiting_announced = False
        self.last_go_announced = False
        self.last_timeout_announced = False

    def detect_crosswalk(self, img_rgb, conf=0.5, cluster_iou=0.2, sum_thresh=0.7, max_thresh=0.7, min_count=2):
        """Detect crosswalk in image using clustering and voting."""
        if self.crosswalk_model is None:
            return False, [], 0.0, 0.0, 0
        
        # Check if this is a YOLO model or ONNX Runtime session
        if not hasattr(self.crosswalk_model, 'predict'):
            # ONNX Runtime session - not directly supported in OutsideEngine
            # The main process_outdoors() function handles ONNX crosswalk detection
            return False, [], 0.0, 0.0, 0
        
        try:
            r = self.crosswalk_model.predict(source=img_rgb, conf=conf, device=0 if torch.cuda.is_available() else "cpu", verbose=False)[0]
        except Exception as e:
            debug_print(f"[Traffic] Crosswalk detection error: {e}")
            return False, [], 0.0, 0.0, 0
        
        if r.boxes is None or len(r.boxes) == 0:
            return False, [], 0.0, 0.0, 0

        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()

        def iou(a, b):
            x1, y1 = max(a[0], b[0]), max(a[1], b[1])
            x2, y2 = min(a[2], b[2]), min(a[3], b[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
            return inter / ua if ua > 0 else 0.0

        clusters = []
        for b, s in zip(boxes, scores):
            for c in clusters:
                if any(iou(b, x) >= cluster_iou for x in c["boxes"]):
                    c["boxes"].append(b)
                    c["scores"].append(float(s))
                    break
            else:
                clusters.append({"boxes": [b], "scores": [float(s)]})

        best = ([], 0.0, 0.0, 0, -1.0)
        for c in clusters:
            ssum, smax, n = sum(c["scores"]), max(c["scores"]), len(c["scores"])
            score = smax + 0.5 * (ssum - smax)
            if score > best[4]:
                best = (c["boxes"], ssum, smax, n, score)

        boxes, ssum, smax, n, _ = best
        detected = (ssum >= sum_thresh and n >= min_count) or (smax >= max_thresh)
        return detected, boxes, ssum, smax, n

    def detect_traffic_light(self, img_rgb, conf_thresh=0.05, top_crop_ratio=0.4, min_square_score=0.9):
        """Detect traffic light state (GO/NO_GO)."""
        if self.traffic_model is None:
            return {"final_go": False, "final_state": "NO_GO", "detections": []}
        
        # Check if this is a YOLO model or ONNX Runtime session
        if not hasattr(self.traffic_model, 'predict'):
            # ONNX Runtime session - not directly supported in OutsideEngine
            return {"final_go": False, "final_state": "NO_GO", "detections": []}
        
        try:
            h, w = img_rgb.shape[:2]
            regions = [
                (img_rgb[:int(h*top_crop_ratio), :w//2], 0),
                (img_rgb[:int(h*top_crop_ratio), w//2:], w//2)
            ]

            traffic_boxes = []
            for img, xoff in regions:
                r = self.traffic_model.predict(source=img, conf=conf_thresh, device=0 if torch.cuda.is_available() else "cpu", verbose=False)[0]
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                xyxy, confs = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()

                for box, conf in zip(xyxy, confs):
                    box = box.copy()
                    box[[0, 2]] += xoff
                    x1, y1, x2, y2 = box
                    bw, bh = x2 - x1, y2 - y1
                    if bw <= 0 or bh <= 0:
                        continue
                    if min(bw, bh) / max(bw, bh) < min_square_score:
                        continue
                    traffic_boxes.append({"box": box, "conf": float(conf)})

            best_green = best_red = 0
            detections = []

            for t in traffic_boxes:
                x1, y1, x2, y2 = map(int, t["box"])
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                crop = img_rgb[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
                white = (hsv[:, :, 1] <= 80) & (hsv[:, :, 2] >= 160)
                crop[white] = [0, 255, 0]
                hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)

                g = cv2.inRange(hsv, (35, 60, 60), (95, 255, 255))
                r1 = cv2.inRange(hsv, (0, 60, 60), (12, 255, 255))
                r2 = cv2.inRange(hsv, (170, 60, 60), (180, 255, 255))
                r = cv2.bitwise_or(r1, r2)

                k = np.ones((3, 3), np.uint8)
                g = cv2.morphologyEx(g, cv2.MORPH_OPEN, k)
                r = cv2.morphologyEx(r, cv2.MORPH_OPEN, k)

                gs, rs = int(np.count_nonzero(g)), int(np.count_nonzero(r))
                state = "GO" if gs > rs and gs > 5 else "NO_GO" if rs > gs and rs > 5 else "UNKNOWN"

                detections.append({
                    "box": [x1, y1, x2, y2],
                    "conf": t["conf"],
                    "green_score": gs,
                    "red_score": rs,
                    "state": state
                })
                best_green, best_red = max(best_green, gs), max(best_red, rs)

            final_go = best_green > best_red and best_green > 5
            return {"final_go": final_go, "final_state": "GO" if final_go else "NO_GO", "detections": detections}
        
        except Exception as e:
            debug_print(f"[Traffic] Traffic light detection error: {e}")
            return {"final_go": False, "final_state": "NO_GO", "detections": []}

    def update_crosswalk_state(self, detected):
        """Update crosswalk state with hysteresis."""
        self.crosswalk_counter += 1 if detected else -1
        if detected and self.crosswalk_counter >= self.crosswalk_hysteresis_on:
            self.crosswalk_state = True
            self.crosswalk_counter = 0
        if not detected and self.crosswalk_counter <= -self.crosswalk_hysteresis_off:
            self.crosswalk_state = False
            self.crosswalk_counter = 0

    def update_traffic_state(self, r):
        """Update traffic state with hysteresis."""
        if r["final_state"] == "GO":
            self.traffic_go_counter += 1
            if self.traffic_go_counter >= self.traffic_hysteresis_go:
                self.traffic_state = "GO"
                self.traffic_go_counter = 0
        else:
            self.traffic_state = r["final_state"]
            self.traffic_go_counter = 0

    def get_status_message(self):
        """Get human-readable status message."""
        return ("Scanning for crosswalk" if not self.crosswalk_state 
                else ("GO" if self.traffic_state == "GO" 
                      else "Waiting for light"))

    def process_frame(self, img_rgb):
        """Process frame and return traffic/crosswalk state."""
        boxes_cw = []
        traffic_result = {"final_go": False, "final_state": "NO_GO", "detections": []}

        # CROSSING MODE: lock system for ~10 seconds after GO signal
        if self.crossing_start_time is not None:
            if time.time() - self.crossing_start_time < self.crossing_hold_time:
                return {
                    "crosswalk_state": True,
                    "crosswalk_boxes": [],
                    "traffic_state": "GO",
                    "traffic_result": traffic_result,
                    "status_message": "GO"
                }
            else:
                self.reset()
                self.crossing_start_time = None

        # PHASE 1: Scan for crosswalk
        if not self.crosswalk_state:
            d, boxes_cw, *_ = self.detect_crosswalk(img_rgb)
            self.update_crosswalk_state(d)
            if d:
                self.last_crosswalk_boxes = boxes_cw
            
            # Announce when crosswalk is first detected
            if self.crosswalk_state and not self.last_crosswalk_announced:
                debug_print("[Traffic] Announcing: Crosswalk detected")
                speak("Crosswalk detected. Please wait for pedestrians to clear, then look for a safe opportunity to cross.")
                self.last_crosswalk_announced = True

        # PHASE 2: Scan for traffic light (if crosswalk detected)
        if self.crosswalk_state:
            # Start timeout timer on first entry
            if self.wait_start_time is None:
                self.wait_start_time = time.time()
            
            traffic_result = self.detect_traffic_light(img_rgb)
            self.update_traffic_state(traffic_result)
            
            # Announce when entering "waiting for light" state
            if self.traffic_state == "NO_GO" and not self.last_waiting_announced:
                debug_print("[Traffic] Announcing: Waiting for light")
                speak("Waiting for traffic signal. Stay alert.")
                self.last_waiting_announced = True
            
            # TIMEOUT: If waiting > 30s, force GO (assume detection failed)
            if self.traffic_state != "GO" and self.wait_start_time is not None:
                if time.time() - self.wait_start_time >= self.wait_timeout:
                    if not self.last_timeout_announced:
                        debug_print(f"[Traffic] Timeout waiting for light (>{self.wait_timeout}s), forcing GO")
                        speak("Light not detected. Proceeding with caution.")
                        self.last_timeout_announced = True
                    self.traffic_state = "GO"

        # Transition to CROSSING MODE when GO detected
        if self.traffic_state == "GO" and self.crossing_start_time is None:
            # Announce GO signal immediately (with interrupt if needed)
            if not self.last_go_announced:
                debug_print("[Traffic] Announcing: GO")
                speak("GO! Proceed across with caution.", interrupt=True)
                self.last_go_announced = True
            self.crossing_start_time = time.time()

        return {
            "crosswalk_state": self.crosswalk_state,
            "crosswalk_boxes": boxes_cw,
            "traffic_state": self.traffic_state,
            "traffic_result": traffic_result,
            "status_message": self.get_status_message()
        }


class OutsideEngine:
    """High-level outdoor navigation combining traffic and people detection."""
    
    def __init__(self, models_dict=None, models_dir="models"):
        """Initialize OutsideEngine with models from dict or directory.
        
        Args:
            models_dict: Dict with 'crosswalk_detect', 'traffic_light_world', 'people_detect' models
            models_dir: Directory to load YOLO models from (used if models_dict is None)
        """
        crosswalk_model = None
        traffic_model = None
        people_model = None
        
        if models_dict:
            # Use provided model dict
            crosswalk_model = models_dict.get("crosswalk_detect")
            traffic_model = models_dict.get("traffic_light_world")
            people_model = models_dict.get("people_detect", None)
        else:
            # Load from models directory using YOLO
            try:
                from ultralytics import YOLO
                models_path = Path(models_dir)
                
                cw_path = models_path / "crosswalk_detect.onnx"
                if cw_path.exists():
                    crosswalk_model = YOLO(str(cw_path), task="detect")
                    debug_print(f"[Traffic] Loaded crosswalk_detect from {cw_path}")
                
                tl_path = models_path / "traffic_light_world.onnx"
                if tl_path.exists():
                    traffic_model = YOLO(str(tl_path), task="detect")
                    debug_print(f"[Traffic] Loaded traffic_light_world from {tl_path}")
                
                pd_path = models_path / "people_detect.onnx"
                if pd_path.exists():
                    people_model = YOLO(str(pd_path), task="detect")
                    debug_print(f"[Traffic] Loaded people_detect from {pd_path}")
                
            except ImportError:
                debug_print("[Traffic] ultralytics not available, YOLO models will not be loaded")
            except Exception as e:
                debug_print(f"[Traffic] Error loading YOLO models: {e}")
        
        self.traffic_engine = CrosswalkTrafficEngine(crosswalk_model, traffic_model)
        self.people_model = people_model
        self.person_class = 0
        
        # Audio announcement tracking for people detection
        self.last_people_announced = False
        self.last_people_count = 0

    def reset(self):
        """Reset all engines."""
        self.traffic_engine.reset()
        self.last_people_announced = False
        self.last_people_count = 0

    def detect_people(self, img_rgb):
        """Detect people in lower portion of frame."""
        if self.people_model is None:
            return []
        
        # Check if this is a YOLO model or ONNX Runtime session
        if not hasattr(self.people_model, 'predict'):
            # ONNX Runtime session - not directly supported in OutsideEngine
            return []
        
        try:
            h, w = img_rgb.shape[:2]
            y = int(h * 0.45)
            roi = img_rgb[y:, :]

            r = self.people_model.predict(
                roi,
                conf=0.7,
                classes=[self.person_class],
                device=0 if torch.cuda.is_available() else "cpu",
                verbose=False
            )[0]

            boxes_out = []
            if r.boxes is not None:
                for box in r.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box.astype(int)
                    if (x2 - x1) * (y2 - y1) < 1200:
                        continue
                    y1 += y
                    y2 += y
                    boxes_out.append([x1, y1, x2, y2])

            return boxes_out
        except Exception as e:
            debug_print(f"[Traffic] People detection error: {e}")
            return []

    def process_frame(self, img_rgb):
        """Process frame with traffic and people detection."""
        traffic_data = self.traffic_engine.process_frame(img_rgb)

        cw = traffic_data["crosswalk_state"]
        tl = traffic_data["traffic_state"]

        people_boxes = []
        people_active = (not cw) and (tl == "UNKNOWN")

        if people_active:
            people_boxes = self.detect_people(img_rgb)
            
            # Announce when people are first detected
            if len(people_boxes) > 0 and not self.last_people_announced:
                debug_print(f"[Traffic] Announcing: People detected ({len(people_boxes)} person/people)")
                speak("People detected nearby. Please wait.", interrupt=False)
                self.last_people_announced = True
                self.last_people_count = len(people_boxes)
            
            # Reset announcement flag when people are no longer detected
            elif len(people_boxes) == 0 and self.last_people_announced:
                self.last_people_announced = False
                self.last_people_count = 0

        return {
            "mode": (
                "PEOPLE" if people_active else
                "TRAFFIC" if cw else
                "CROSSWALK"
            ),
            "crosswalk_state": cw,
            "crosswalk_boxes": traffic_data["crosswalk_boxes"],
            "traffic_state": tl,
            "traffic_result": traffic_data["traffic_result"],
            "people_boxes": people_boxes,
            "people_count": len(people_boxes),
            "crosswalk_boxes": traffic_data["crosswalk_boxes"],
            "status_message": self._status(cw, tl, len(people_boxes))
        }

    def _status(self, cw, tl, people_count):
        """Generate status message."""
        if not cw:
            return "Scanning for crosswalk"
        if tl == "GO":
            return "GO"
        if tl != "UNKNOWN":
            return "Waiting for light"
        if people_count > 0:
            return "People detected"
        return "Idle"


def _initialize_ray_casting():
    global _ray_config, _ray_caster, _corridor_finder, _steering_cmd
    
    if _ray_config is None:
        _ray_config = RayConfig(
            n_rays=65,
            fov_deg=110.0,    
            max_range_frac=5,
            vertical_scale=0.10,
            min_obstacle_area=2000  
        )
        _ray_caster = RayCaster(_ray_config)
        _corridor_finder = CorridorFinder(win_size=11, center_bias=0.7)
        _steering_cmd = SteeringCommand(frame_buffer_size=40, inertia_weight=20)

def process_cw(output, orig_w, orig_h, conf_thresh=0.65, iou_thresh=0.5):
    """Postprocess YOLOv8 ONNX output for crosswalk detection.
    
    Args:
        output: Raw model output
        orig_w: Original frame width
        orig_h: Original frame height
        conf_thresh: Confidence threshold
        iou_thresh: IOU threshold for NMS
        
    Returns:
        List of detected boxes [(x1, y1, x2, y2, confidence), ...]
    """
    out = np.squeeze(output)  # (5, 5376) or similar shape

    x = out[0]
    y = out[1]
    w = out[2]
    h = out[3]
    conf = out[4]

    boxes = []
    for i in range(out.shape[1]):
        if conf[i] < conf_thresh:
            continue

        # XYWH in YOLO input space
        x1 = x[i] - w[i] / 2
        y1 = y[i] - h[i] / 2
        x2 = x[i] + w[i] / 2
        y2 = y[i] + h[i] / 2

        # Scale to original image size
        x1 = int(x1 * orig_w / YOLO_INPUT_W)
        y1 = int(y1 * orig_h / YOLO_INPUT_H)
        x2 = int(x2 * orig_w / YOLO_INPUT_W)
        y2 = int(y2 * orig_h / YOLO_INPUT_H)

        # Clip to image bounds
        x1 = max(0, min(x1, orig_w - 1))
        y1 = max(0, min(y1, orig_h - 1))
        x2 = max(0, min(x2, orig_w - 1))
        y2 = max(0, min(y2, orig_h - 1))

        boxes.append((x1, y1, x2, y2, float(conf[i])))

    # Non-maximum suppression (NMS) to merge close boxes
    def compute_iou(box1, box2):
        x1_i = max(box1[0], box2[0])
        y1_i = max(box1[1], box2[1])
        x2_i = min(box1[2], box2[2])
        y2_i = min(box1[3], box2[3])
        
        inter_area = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0

    # Sort by confidence descending
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    
    # Apply NMS
    merged = []
    used = set()
    
    for i, box in enumerate(boxes):
        if i in used:
            continue
        merged.append(box)
        
        for j in range(i + 1, len(boxes)):
            if j in used:
                continue
            if compute_iou(box, boxes[j]) > iou_thresh:
                used.add(j)
    
    return merged

def process_outdoors(outdoor_model, frame, crosswalk_model=None):
    global _θ_cmd_smooth
    
    _initialize_ray_casting()
    
    if outdoor_model is None:
        debug_print("[AI][OUTDOOR] Model not loaded")
        return None, False
    
    street_detected = False
    
    try:
        # Preprocess frame for obstacle detection
        outFrame = preprocess(frame, pp_type='BI-SEG')
        
        # Get output name from model
        output_name = outdoor_model.get_outputs()[0].name
        result = outdoor_model.run([output_name], {'input': outFrame})
        mask = result[0]
        
        # Apply sigmoid and threshold
        mask_sigmoid = torch.sigmoid(torch.from_numpy(mask)).squeeze().numpy()
        binary_mask = (mask_sigmoid > 0.5).astype(float)
        
        # Resize mask to match original frame size
        frame_h, frame_w = frame.shape[:2]
        mask_resized = cv2.resize(binary_mask, (frame_w, frame_h))
        
        # Convert to 8-bit for ray casting (0-255 range)
        mask_8bit = (mask_resized * 255).astype(np.uint8)
        
        # Ray casting for navigation
        origin = (frame_w // 2, frame_h - 1)
        distances = _ray_caster.cast_rays(mask_8bit, origin)
        
        # Find best corridor
        hann_weights = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(_corridor_finder.win_size) / 
                                           (_corridor_finder.win_size - 1))
        scores, best_corridor, θ_best, confidence = _corridor_finder.sliding_window(
            distances, _ray_config.angles, hann_weights
        )
        
        # Smooth steering angle
        θ_cmd = θ_best * (0.5 + 0.5 * confidence)
        smoothing_alpha = 0.8
        _θ_cmd_smooth = smoothing_alpha * θ_cmd + (1 - smoothing_alpha) * _θ_cmd_smooth
        
        # Compute steering command
        cmd = _steering_cmd.compute(_θ_cmd_smooth)
        
        # Crosswalk detection if model is available
        if crosswalk_model is not None:
            try:
                cw_frame = preprocess(frame, pp_type='YOLO CW')
                cw_output_name = crosswalk_model.get_outputs()[0].name
                cw_result = crosswalk_model.run([cw_output_name], {'images': cw_frame})
                
                # Process crosswalk detections with higher confidence threshold
                detections = process_cw(cw_result[0], frame_w, frame_h, conf_thresh=0.55)
                
                # Filter detections to only those in the lower portion of frame (near path)
                # With majority voting, just check if bottom of detection is in lower half of frame
                relevant_detections = []
                for x1, y1, x2, y2, conf in detections:
                    # Check if detection is in lower half of frame
                    if y2 > frame_h * 0.5:
                        relevant_detections.append((x1, y1, x2, y2, conf))
                
                # Add detection result to buffer for majority voting
                frame_street_detected = len(relevant_detections) > 0
                _crosswalk_buffer.append(frame_street_detected)
                if len(_crosswalk_buffer) > _crosswalk_buffer_size:
                    _crosswalk_buffer.pop(0)
                
                # Majority vote: True if more than half the buffer is True
                street_detected = sum(_crosswalk_buffer) > len(_crosswalk_buffer) / 2
                
            except Exception as e:
                debug_print(f"[AI][OUTDOOR] Crosswalk detection error: {e}")
                # Add False to buffer on error
                _crosswalk_buffer.append(False)
                if len(_crosswalk_buffer) > _crosswalk_buffer_size:
                    _crosswalk_buffer.pop(0)
                street_detected = sum(_crosswalk_buffer) > len(_crosswalk_buffer) / 2
        else:
            street_detected = False

        return cmd, street_detected
        
    except Exception as e:
        debug_print(f"[AI][OUTDOOR] Error during processing: {e}")
        return None, False

def init_outside_engine(models_dict=None, models_dir="models"):
    """Initialize the OutsideEngine with preloaded models or from directory."""
    global _outside_engine
    
    if _outside_engine is None:
        try:
            _outside_engine = OutsideEngine(models_dict=models_dict, models_dir=models_dir)
            debug_print("[Traffic] OutsideEngine initialized successfully")
        except Exception as e:
            debug_print(f"[Traffic] Failed to initialize OutsideEngine: {e}")
            _outside_engine = None
    
    return _outside_engine


def get_outside_engine():
    """Get the current OutsideEngine instance."""
    return _outside_engine
