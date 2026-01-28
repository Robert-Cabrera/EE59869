import numpy as np
from collections import deque
from src.image_pp import preprocess

FORWARD_BIAS = 0.3
SMOOTHED_DX = 0.0
LAST_CMD = "FORWARD"
ALPHA = 0.3
HYST = 0.05

BUFFER_SIZE = 10
cmd_buffer = deque(maxlen=BUFFER_SIZE)

DIRECTION_THRESHOLDS = {
    "LEFT": -0.3,
    "SLIGHT_LEFT": -0.1,
    "FORWARD": 0.1,
    "SLIGHT_RIGHT": 0.3,
    "RIGHT": 0.5
}

COMMAND_STABILITY_BUFFER = deque(maxlen=20)
MIN_STABILITY_FRAMES = 15


def compute_best_direction(depth_norm):
    def compute_confidence(depth_norm, col_scores, best_col):
        H, W = depth_norm.shape
        roi = depth_norm[int(H * 0.60):, :]
        depth_variance = np.var(roi)
        variance_confidence = np.clip(depth_variance * 10, 0, 1)
        
        sorted_scores = np.sort(col_scores)
        if len(sorted_scores) > 1:
            best_score = sorted_scores[-1]
            second_best = sorted_scores[-2]
            mean_score = np.mean(col_scores)
            separation_confidence = np.clip((best_score - second_best) / mean_score * 2, 0, 1) if mean_score > 1e-6 else 0.0
        else:
            separation_confidence = 0.0
        
        score_sum = np.sum(col_scores)
        if score_sum > 1e-6:
            probs = col_scores / score_sum
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = np.log(len(col_scores))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 1.0
            entropy_confidence = 1.0 - normalized_entropy
        else:
            entropy_confidence = 0.0
        
        center = (W - 1) / 2
        distance_from_center = abs(best_col - center) / center
        centrality_confidence = 1.0 - distance_from_center
        
        confidence = (
            0.30 * variance_confidence +
            0.35 * separation_confidence +
            0.25 * entropy_confidence +
            0.10 * centrality_confidence
        )
        return np.clip(confidence, 0.0, 1.0)

    H, W = depth_norm.shape
    roi = depth_norm[int(H * 0.60):, :]
    roi_far = 1.0 - roi
    col_scores = roi_far.mean(axis=0)
    
    if FORWARD_BIAS > 0:
        center = (W - 1) / 2
        col_indices = np.arange(W)
        sigma = W * (1.0 - FORWARD_BIAS)
        gaussian_weights = np.exp(-((col_indices - center) ** 2) / (2 * sigma ** 2))
        col_scores = col_scores * (1.0 + FORWARD_BIAS * gaussian_weights)
    
    best_col = np.argmax(col_scores)
    center = (W - 1) / 2
    confidence = compute_confidence(depth_norm, col_scores, best_col)
    
    dx = (best_col - center) / center
    dy = -1
    mag = np.sqrt(dx*dx + dy*dy)
    return dx/mag, dy/mag, confidence


def discretize_direction(dx):
    global LAST_CMD, SMOOTHED_DX
    T = DIRECTION_THRESHOLDS
    
    SMOOTHED_DX = ALPHA * dx + (1 - ALPHA) * SMOOTHED_DX
    filtered_dx = SMOOTHED_DX
    
    candidate_cmd = None
    if LAST_CMD == "LEFT" and filtered_dx < T["LEFT"] + HYST:
        candidate_cmd = LAST_CMD
    elif LAST_CMD == "SLIGHT_LEFT" and T["LEFT"] + HYST <= filtered_dx < T["SLIGHT_LEFT"] + HYST:
        candidate_cmd = LAST_CMD
    elif LAST_CMD == "FORWARD" and T["SLIGHT_LEFT"] + HYST <= filtered_dx < T["FORWARD"] + HYST:
        candidate_cmd = LAST_CMD
    elif LAST_CMD == "SLIGHT_RIGHT" and T["FORWARD"] + HYST <= filtered_dx < T["SLIGHT_RIGHT"] + HYST:
        candidate_cmd = LAST_CMD
    elif LAST_CMD == "RIGHT" and filtered_dx > T["SLIGHT_RIGHT"] - HYST:
        candidate_cmd = LAST_CMD
    else:
        if filtered_dx < T["LEFT"]:
            candidate_cmd = "LEFT"
        elif filtered_dx < T["SLIGHT_LEFT"]:
            candidate_cmd = "SLIGHT_LEFT"
        elif filtered_dx < T["FORWARD"]:
            candidate_cmd = "FORWARD"
        elif filtered_dx < T["SLIGHT_RIGHT"]:
            candidate_cmd = "SLIGHT_RIGHT"
        else:
            candidate_cmd = "RIGHT"
    
    cmd_buffer.append(candidate_cmd)
    if len(cmd_buffer) < BUFFER_SIZE:
        return LAST_CMD
    
    vote_counts = {}
    for cmd in cmd_buffer:
        vote_counts[cmd] = vote_counts.get(cmd, 0) + 1
    
    majority_cmd = max(vote_counts, key=vote_counts.get)
    LAST_CMD = majority_cmd
    return LAST_CMD


def process_indoors(indoor_model, frame):
    if indoor_model is None:
        print("[AI][INDOOR] Model not loaded")
        return None, None, None
    
    indoorFrame = preprocess(frame, pp_type='MIDAS')
    depth = indoor_model.run(None, {'input_image': indoorFrame})
    depth_image = depth[0][0]
    depth_norm = depth_image / depth_image.max()
    
    dx,_,_ = compute_best_direction(depth_norm)
    command = discretize_direction(dx)
    
    return command
