import time
import threading
import board
import busio
import adafruit_vl53l1x
import pin_util as pu
from collections import deque

leftXSHUT   = 32
centerXSHUT = 31
rightXSHUT  = 33

# left->0x30, center->0x31, right stays 0x29
xshut_pins = [leftXSHUT, centerXSHUT, rightXSHUT]
new_addrs  = [0x30, 0x31, 0x29]
addr_names = ["left", "center", "right"]

BOOT_DELAY_S   = 0.30
INIT_RETRIES   = 10
RETRY_DELAY_S  = 0.12

class ToFThread:
    def __init__(self, buffer_size=10):
        self.buffer_size = buffer_size
        self.buffers = {
            "left": deque(maxlen=buffer_size),
            "center": deque(maxlen=buffer_size),
            "right": deque(maxlen=buffer_size)
        }
        self.averages = {
            "left": 0,
            "center": 0,
            "right": 0
        }
        self.sensors = []
        self.thread = None
        self.running = False
        self.lock = threading.Lock()
        
    def make_sensor_default(self, i2c):
        """Create sensor at default address."""
        last_err = None
        for _ in range(INIT_RETRIES):
            try:
                return adafruit_vl53l1x.VL53L1X(i2c)  # default 0x29
            except RuntimeError as e:
                last_err = e
                time.sleep(RETRY_DELAY_S)
        raise last_err

    def make_sensor_at(self, i2c, addr):
        """Create sensor at specified address."""
        last_err = None
        for _ in range(INIT_RETRIES):
            try:
                return adafruit_vl53l1x.VL53L1X(i2c, address=addr)
            except RuntimeError as e:
                last_err = e
                time.sleep(RETRY_DELAY_S)
        raise last_err

    def all_off(self):
        """Set all XSHUT pins low."""
        for p in xshut_pins:
            pu.low(p)

    def initialize_sensors(self):
        """Initialize all ToF sensors."""
        try:
            pu.gpio_init()
            for p in xshut_pins:
                pu.setup_out(p)

            # Hard reset once
            self.all_off()
            time.sleep(0.25)

            # Use explicit bus + lower freq for stability
            i2c = busio.I2C(board.SCL, board.SDA, frequency=100000)

            # Bring up one-by-one
            for pin, addr in zip(xshut_pins, new_addrs):
                pu.high(pin)
                time.sleep(BOOT_DELAY_S)

                # At this point, only this newly-enabled sensor is at 0x29
                s = self.make_sensor_default(i2c)

                if addr != 0x29:
                    s.set_address(addr)
                    time.sleep(0.05)
                    # Rebind object to new address
                    s = self.make_sensor_at(i2c, addr)

                self.sensors.append(s)

            # Verify bus
            if i2c.try_lock():
                print("[ToF] I2C scan:", [hex(x) for x in i2c.scan()])
                i2c.unlock()

            # Start ranging
            for s in self.sensors:
                s.start_ranging()
                time.sleep(0.02)
                s.clear_interrupt()

            print("[ToF] Sensors initialized successfully")
            
        except Exception as e:
            print(f"[ToF] Initialization error: {e}")
            raise

    def run(self):
        """Main thread loop to read sensors and populate buffers."""
        try:
            self.initialize_sensors()
            self.running = True
            
            while self.running:
                for idx, s in enumerate(self.sensors):
                    side = addr_names[idx]
                    
                    if s.data_ready:
                        distance = s.distance
                        
                        # Add to buffer only if distance is valid
                        if distance is not None:
                            with self.lock:
                                self.buffers[side].append(distance)
                                # Update average
                                if len(self.buffers[side]) > 0:
                                    self.averages[side] = sum(self.buffers[side]) / len(self.buffers[side])
                        
                        s.clear_interrupt()
                
                time.sleep(0.02)  # Non-blocking read interval
                
        except Exception as e:
            print(f"[ToF] Thread error: {e}")
        finally:
            self.cleanup()

    def get_averages(self):
        """Get current averaged distances from all sensors."""
        with self.lock:
            return {k: round(v, 2) for k, v in self.averages.items()}

    def get_average(self, side):
        """Get current averaged distance for a specific sensor."""
        with self.lock:
            return round(self.averages.get(side, 0), 2)

    def start(self):
        """Start the ToF thread."""
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
            print("[ToF] Thread started")

    def stop(self):
        """Stop the ToF thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        print("[ToF] Thread stopped")

    def cleanup(self):
        """Clean up resources."""
        try:
            self.all_off()
        except Exception:
            pass
        try:
            pu.cleanup()
        except Exception:
            pass


# Global instance
_tof_thread = None

def init_tof(buffer_size=10):
    """Initialize and start the ToF thread."""
    global _tof_thread
    _tof_thread = ToFThread(buffer_size=buffer_size)
    _tof_thread.start()
    return _tof_thread

def get_tof_thread():
    """Get the global ToF thread instance."""
    return _tof_thread

def get_distances():
    """Get current averaged distances from all sensors."""
    if _tof_thread:
        return _tof_thread.get_averages()
    return {"left": 0, "center": 0, "right": 0}

def cleanup_tof():
    """Stop the ToF thread and clean up."""
    global _tof_thread
    if _tof_thread:
        _tof_thread.stop()
        _tof_thread = None
