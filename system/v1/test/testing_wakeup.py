import os
import time
import subprocess
from datetime import datetime

LOG_FILE = "resume_test.log"

def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(message)

def main():
    # Clear previous log
    os.system('rm -f ' + LOG_FILE)  
    log_message("--- Starting Sleep Test ---")
    
    for i in range(5, 0, -1):
        log_message(f"Going to sleep in {i} seconds...")
        time.sleep(1)

    log_message("TRIGGERING DEEP SLEEP NOW.")
    
    # After this line, the Jetson powers down and we should resume when shorting J14 POWER ON and GROUND
    try:
        # Use 'mem' state (standard memory/suspend state)
        subprocess.run(['sudo', 'bash', '-c', 'echo mem > /sys/power/state'], check=True)
    except subprocess.CalledProcessError as e:
        log_message(f"Error triggering sleep: {e}")

    # We should resume right here

    log_message("WOKE UP! Resume successful.")
    log_message("--- Test Complete ---")

if __name__ == "__main__":
    main()