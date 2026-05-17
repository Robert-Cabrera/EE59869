"""
Frame recording module for saving frames with arrows during execution.
Uses the session directory created by debug_config.
"""

import os
from datetime import datetime
from pathlib import Path
import cv2
from .debug_config import debug_print, get_session_dir, init_session

class FrameRecorder:
    """Records frames with arrow overlays to session folder."""
    
    def __init__(self):
        self.enabled = False
        self.frames_dir = None
        self.frame_count = 0
    
    def start_session(self):
        """Initialize frame recording in the current session."""
        if not self.enabled:
            return
        
        # Get the session directory (should have been created by init_session() in main)
        session_dir = get_session_dir()
        
        # Fallback: if session hasn't been initialized yet, do it now
        if session_dir is None:
            session_dir = init_session()
        
        if session_dir is None:
            debug_print("[Recording] ERROR: Could not initialize session directory")
            return
        
        # Create frames subdirectory
        self.frames_dir = session_dir / f"{session_dir.name}_frames"
        self.frames_dir.mkdir(exist_ok=True)
        
        debug_print(f"[Recording] Frame recording enabled for session: {session_dir.name}")
    
    def stop_session(self):
        """Close the recording session."""
        if self.enabled and self.frames_dir:
            pass  # Silent - no logging
    
    def save_frame(self, frame, command, additional_info=""):
        """Save a frame with metadata."""
        if not self.enabled or self.frames_dir is None:
            return
        
        frame_path = self.frames_dir / f"frame_{self.frame_count:06d}.jpg"
        
        # Save frame
        cv2.imwrite(str(frame_path), frame)
        
        self.frame_count += 1
    
    def enable(self):
        """Enable frame recording."""
        self.enabled = True
    
    def is_enabled(self):
        """Check if frame recording is enabled."""
        return self.enabled
    
    def get_session_path(self):
        """Get the frames directory path."""
        return self.frames_dir


# Global recorder instance
_recorder = FrameRecorder()

def get_recorder():
    """Get the global frame recorder instance."""
    return _recorder
