"""
Centralized debug configuration for all modules.
Controls whether debug prints are enabled throughout the system.
"""

import os
from datetime import datetime
from pathlib import Path

# Global debug flag - enables console output
DEBUG = False

# Session directory - always created to store logs
SESSION_DIR = None

# Set up logs parent directory
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

def init_session() -> Path:
    """
    Initialize a session directory for logging.
    This is ALWAYS called at startup to create the logs folder structure.
    
    Returns:
        Path to the session directory
    """
    global SESSION_DIR
    
    # Create timestamp for this session
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create session directory
    SESSION_DIR = LOGS_DIR / timestamp
    SESSION_DIR.mkdir(exist_ok=True)
    
    return SESSION_DIR

def get_session_dir() -> Path:
    """Get the current session directory."""
    return SESSION_DIR

def set_debug(enabled: bool) -> None:
    """
    Enable or disable debug printing to console.
    
    Args:
        enabled (bool): True to enable console output, False to disable
    """
    global DEBUG
    DEBUG = enabled

def debug_print(*args, **kwargs) -> None:
    """
    Always logs messages to the session log file.
    Prints to console only if DEBUG flag is enabled.
    
    Args:
        *args: Arguments to pass to print()
        **kwargs: Keyword arguments to pass to print()
    """
    # Format message
    message = " ".join(str(arg) for arg in args)
    
    # Get current timestamp
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    # Always write to session log file
    if SESSION_DIR:
        log_file = SESSION_DIR / f"{SESSION_DIR.name}_log.txt"
        try:
            with open(log_file, "a") as f:
                f.write(f"[{timestamp}] {message}\n")
        except Exception as e:
            # Silently fail if we can't write to log
            pass
    
    # Print to console only if DEBUG is enabled
    if DEBUG:
        print(*args, **kwargs)

def is_debug_enabled() -> bool:
    """
    Check if console output is currently enabled.
    
    Returns:
        bool: True if DEBUG is enabled, False otherwise
    """
    return DEBUG
