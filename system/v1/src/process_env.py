import cv2
import numpy as np
import torch
import clip
from PIL import Image
from collections import deque

# Internal state for environment processing
_frame_buffer = []
_env_counter = 0
_buffer_size = 10
_clip_preprocessor = None
_text_tokens = None
_prediction_buffer = deque(maxlen=5)  # Buffer last 5 predictions for voting
_last_voted_result = "unknown"
_initialized = False


def _initialize_clip():
    """Initialize CLIP preprocessing and text tokens once"""
    global _clip_preprocessor, _text_tokens, _initialized
    
    if not _initialized:
        try:
            # Load CLIP preprocessing pipeline
            _, _clip_preprocessor = clip.load("RN50", device="cpu")
            # Pre-tokenize labels as int64
            _text_tokens = clip.tokenize(["indoor", "outdoor"]).numpy().astype(np.int64)
            _initialized = True
        except Exception as e:
            print(f"[AI][ENV] Error initializing CLIP: {e}")
            return False
    return True


def process_env(env_model, frame):
    """Process frame to determine if environment is indoor or outdoor
    
    Args:
        env_model: ONNX session for CLIP model
        frame: Input frame (BGR)
        
    Returns:
        str: "indoor", "outdoor", or "unknown"
    """
    global _frame_buffer, _env_counter, _prediction_buffer, _last_voted_result
    
    if env_model is None:
        print("[AI][ENV] Model not loaded")
        return "unknown"
    
    if frame is None:
        print("[AI][ENV] Error during inference: Received None frame")
        return "unknown"
    
    if not _initialize_clip():
        return "unknown"
    
    # Add frame to buffer
    _frame_buffer.append(frame)
    _env_counter += 1
    
    # Check if buffer is full (every 10 frames)
    if len(_frame_buffer) >= _buffer_size:
        try:
            # Convert to PIL
            img_pil = Image.fromarray(cv2.cvtColor(_frame_buffer[0], cv2.COLOR_BGR2RGB))
            
            # Preprocess → (1, 3, 224, 224) FP32
            img = _clip_preprocessor(img_pil).unsqueeze(0).numpy()
            
            # ONNX forward pass
            outputs = env_model.run(
                ["logits_per_image"],
                {
                    "image": img.astype(np.float32),
                    "text": _text_tokens  # int64
                }
            )[0][0]
            
            # Apply softmax to get probabilities
            probs = torch.softmax(torch.tensor(outputs), dim=-1).numpy()
            
            # Determine prediction
            current_prediction = "indoor" if probs[0] > probs[1] else "outdoor"
            
            # Add to voting buffer
            _prediction_buffer.append(current_prediction)
            
            # Majority voting
            if len(_prediction_buffer) >= 3:
                vote_counts = {}
                for pred in _prediction_buffer:
                    vote_counts[pred] = vote_counts.get(pred, 0) + 1
                
                _last_voted_result = max(vote_counts, key=vote_counts.get)
            else:
                _last_voted_result = current_prediction
            
            print(f"[AI][ENV] Prediction refreshed: {_last_voted_result}...")
            
            # Clear buffer
            _frame_buffer = []
            _env_counter = 0
            
            return _last_voted_result
            
        except Exception as e:
            print(f"[AI][ENV] Error during inference: {e}")
            return "unknown"
    
    # Buffer not full yet, return last cached result
    return _last_voted_result
