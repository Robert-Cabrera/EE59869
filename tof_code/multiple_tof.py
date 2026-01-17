import time
import board
import busio
import adafruit_vl53l1x
import pin_util as pu

leftXSHUT   = 32
centerXSHUT = 31
rightXSHUT  = 33

# left->0x30, center->0x31, right stays 0x29
xshut_pins = [leftXSHUT, centerXSHUT, rightXSHUT]
new_addrs  = [0x30, 0x31, 0x29]   

BOOT_DELAY_S   = 0.30
INIT_RETRIES   = 10
RETRY_DELAY_S  = 0.12

def make_sensor_default(i2c):
    last_err = None
    for _ in range(INIT_RETRIES):
        try:
            return adafruit_vl53l1x.VL53L1X(i2c)  # default 0x29
        except RuntimeError as e:
            last_err = e
            time.sleep(RETRY_DELAY_S)
    raise last_err

def make_sensor_at(i2c, addr):
    last_err = None
    for _ in range(INIT_RETRIES):
        try:
            return adafruit_vl53l1x.VL53L1X(i2c, address=addr)
        except RuntimeError as e:
            last_err = e
            time.sleep(RETRY_DELAY_S)
    raise last_err

def all_off():
    for p in xshut_pins:
        pu.low(p)
        
def address_to_side(addr):
    if addr == 0x30:
        return "left"
    elif addr == 0x31:
        return "center"
    elif addr == 0x29:
        return "right"
    else:
        return "unknown"

pu.gpio_init()
for p in xshut_pins:
    pu.setup_out(p)

# Hard reset once
all_off()
time.sleep(0.25)

# Use explicit bus + lower freq for stability
i2c = busio.I2C(board.SCL, board.SDA, frequency=100000)

sensors = []

try:
    # Bring up one-by-one, but KEEP previous sensors ON after they get new addresses.
    for pin, addr in zip(xshut_pins, new_addrs):
        pu.high(pin)
        time.sleep(BOOT_DELAY_S)

        # At this point, all previously-enabled sensors already have unique addresses,
        # and ONLY this newly-enabled one is still at 0x29.
        s = make_sensor_default(i2c)

        if addr != 0x29:
            s.set_address(addr)
            time.sleep(0.05)
            # Rebind object to new address
            s = make_sensor_at(i2c, addr)

        sensors.append(s)

    # Verify bus
    if i2c.try_lock():
        print("I2C scan:", [hex(x) for x in i2c.scan()])
        i2c.unlock()

    # Start ranging
    for s in sensors:
        s.start_ranging()
        time.sleep(0.02)
        s.clear_interrupt()

    while True:
        for idx, s in enumerate(sensors):
            addr = new_addrs[idx]
            if s.data_ready:
                d = s.distance
                st = getattr(s, "range_status", "GOOD")
                print(f"[STATUS = {st}] {address_to_side(addr)}:{d}")
                s.clear_interrupt()
        print()
        time.sleep(0.2)

except KeyboardInterrupt:
    print("Exiting...")

finally:
    # Drive XSHUT low so next run starts clean
    try:
        all_off()
    except Exception:
        pass
    pu.cleanup()
