import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


def preprocess(frame, pp_type="ENV"):
    """Preprocess frame for different models
    
    Args:
        frame: Input frame (BGR)
        pp_type: Type of preprocessing - "ENV", "MIDAS", "BI-SEG", "YOLO CW"
        
    Returns:
        Preprocessed frame as numpy array
    """
    
    # Define normalization for BI-SEG
    IMAGE_NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
    
    match pp_type:
        case "ENV":
            return frame
        
        case "MIDAS":
            size = (256, 256)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))[None, ...]
            return img
        
        case "BI-SEG":
            # Convert BGR to RGB and then to PIL
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # Resize to (256, 256)
            img_resized = T.Resize((256, 256))(img_pil)
            
            # Convert to tensor (divides by 255)
            img_tensor = T.ToTensor()(img_resized)
            
            # Apply normalization
            img_normalized = IMAGE_NORMALIZE(img_tensor)
            
            # Add batch dimension and convert to numpy
            img_np = img_normalized.unsqueeze(0).numpy()
            
            return img_np.astype(np.float32)
        
        case "YOLO CW":
            # Preprocess for YOLO crosswalk detection
            resized = cv2.resize(frame, (512, 512))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            norm = rgb / 255.0
            blob = np.transpose(norm, (2, 0, 1)).astype(np.float32)
            return blob[np.newaxis, :, :, :]

    return None