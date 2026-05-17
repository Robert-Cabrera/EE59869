#!/usr/bin/env python3
"""
Host tools for viewing recorded frame sequences from debug sessions.
Simple terminal-based menu for selecting and playback.
"""

import os
import cv2
import sys
from pathlib import Path

def get_sessions():
    """Get list of all recorded session directories."""
    logs_dir = Path(__file__).parent / "logs"
    if not logs_dir.exists():
        return []
    
    sessions = []
    for item in sorted(logs_dir.iterdir(), reverse=True):
        if item.is_dir() and "_" in item.name:
            # Check if there's a matching frames folder
            frames_dir = item / f"{item.name}_frames"
            if frames_dir.exists():
                sessions.append(item)
    
    return sessions

def get_log_info(session_dir):
    """Get session log information."""
    log_file = session_dir / f"{session_dir.name}_log.txt"
    if not log_file.exists():
        return []
    
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
        return lines
    except:
        return []

def get_frame_files(session_dir):
    """Get all frame files from a session."""
    frames_dir = session_dir / f"{session_dir.name}_frames"
    if not frames_dir.exists():
        return []
    
    frames = sorted([f for f in frames_dir.glob("frame_*.jpg")])
    return frames

def play_session(session_dir, fps=10):
    """Play frames from a session as video."""
    frames = get_frame_files(session_dir)
    if not frames:
        print(f"No frames found in {session_dir}")
        return
    
    print(f"\nPlaying {len(frames)} frames from {session_dir.name}")
    print(f"FPS: {fps} | Press 'q' to quit, 'p' to pause/resume, 'd' for slower, 'f' for faster")
    print("-" * 80)
    
    paused = False
    current_frame = 0
    delay = int(1000 / fps)  # milliseconds
    
    while current_frame < len(frames):
        frame_path = frames[current_frame]
        frame = cv2.imread(str(frame_path))
        
        if frame is None:
            current_frame += 1
            continue
        
        # Add frame info text
        frame_text = f"Frame {current_frame + 1}/{len(frames)}"
        cv2.putText(frame, frame_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)
        
        cv2.imshow("Session Playback", frame)
        
        key = cv2.waitKey(delay if not paused else 0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('d'):
            fps = max(1, fps - 1)
            delay = int(1000 / fps)
        elif key == ord('f'):
            fps = min(60, fps + 1)
            delay = int(1000 / fps)
        elif not paused:
            current_frame += 1
    
    cv2.destroyAllWindows()

def display_sessions(sessions, selected):
    """Display the session list menu."""
    os.system('clear' if os.name == 'posix' else 'cls')
    
    print("=" * 80)
    print("Recording Session Viewer")
    print("=" * 80)
    print(f"\nAvailable Sessions: {len(sessions)}\n")
    
    for i, session in enumerate(sessions):
        frames = get_frame_files(session)
        frame_count = len(frames)
        prefix = "➤ " if i == selected else "  "
        
        print(f"{prefix}[{i}] {session.name} ({frame_count} frames)")
    
    print("\n" + "=" * 80)
    print("Commands: <number> = select session, 'l' = view logs, 'p' = play, 'q' = quit")
    print("=" * 80)

def display_logs(session):
    """Display logs for a session."""
    os.system('clear' if os.name == 'posix' else 'cls')
    
    log_lines = get_log_info(session)
    
    print("=" * 80)
    print(f"Logs for: {session.name}")
    print("=" * 80)
    print()
    
    # Show all log lines with scrolling support if needed
    for i, line in enumerate(log_lines):
        print(line.strip())
        # If lots of logs, show them in chunks
        if (i + 1) % 30 == 0 and i < len(log_lines) - 1:
            print("\n(Press Enter to continue...)")
            try:
                input()
                os.system('clear' if os.name == 'posix' else 'cls')
                print("=" * 80)
                print(f"Logs for: {session.name} (continued)")
                print("=" * 80)
                print()
            except:
                pass
    
    print("\n" + "=" * 80)

def main():
    """Main entry point."""
    sessions = get_sessions()
    
    if not sessions:
        print("No recording sessions found!")
        print("\nRun main.py with --record flag to create session.")
        print("Example: python main.py --record --debug")
        return
    
    selected = 0
    
    while True:
        display_sessions(sessions, selected)
        
        try:
            user_input = input("\nEnter command: ").strip().lower()
            
            if user_input == 'q':
                print("Goodbye!")
                break
            elif user_input == 'p':
                play_session(sessions[selected])
            elif user_input == 'l':
                display_logs(sessions[selected])
                input("\nPress Enter to go back...")
            elif user_input.isdigit():
                idx = int(user_input)
                if 0 <= idx < len(sessions):
                    selected = idx
                else:
                    print(f"Invalid session number. Please enter 0-{len(sessions)-1}")
            else:
                print("Invalid command. Please try again.")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()