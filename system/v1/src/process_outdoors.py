import cv2
import numpy as np
import torch
from collections import Counter
from src.image_pp import preprocess

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
_crosswalk_buffer_size = 10  # Number of frames to keep for majority voting


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
    def __init__(self, frame_buffer_size=40, inertia_weight=20):
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

def _initialize_ray_casting():
    global _ray_config, _ray_caster, _corridor_finder, _steering_cmd
    
    if _ray_config is None:
        _ray_config = RayConfig(
            n_rays=50,
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
        print("[AI][OUTDOOR] Model not loaded")
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
                detections = process_cw(cw_result[0], frame_w, frame_h, conf_thresh=0.85)
                
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
                print(f"[AI][OUTDOOR] Crosswalk detection error: {e}")
                # Add False to buffer on error
                _crosswalk_buffer.append(False)
                if len(_crosswalk_buffer) > _crosswalk_buffer_size:
                    _crosswalk_buffer.pop(0)
                street_detected = sum(_crosswalk_buffer) > len(_crosswalk_buffer) / 2
        else:
            street_detected = False

        return cmd, street_detected
        
    except Exception as e:
        print(f"[AI][OUTDOOR] Error during processing: {e}")
        return None, False