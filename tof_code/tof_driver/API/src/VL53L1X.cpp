#include "VL53L1X.hpp"
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <unistd.h>
#include <thread>

using namespace std::chrono_literals;

VL53L1X::VL53L1X(int fd, uint8_t addr) : fd(fd), address(addr), interruptPolarity(0) {}

void VL53L1X::write8Reg16(uint8_t addr, uint16_t reg, uint8_t val) {
    ioctl(this->fd, I2C_SLAVE, addr);
    uint8_t buf[3] = {(uint8_t)(reg >> 8), (uint8_t)(reg & 0xFF), val};
    write(this->fd, buf, 3);
}

void VL53L1X::write16Reg16(uint8_t addr, uint16_t reg, uint16_t val) {
    ioctl(this->fd, I2C_SLAVE, addr);
    uint8_t buf[4] = {(uint8_t)(reg >> 8), (uint8_t)(reg & 0xFF), (uint8_t)(val >> 8), (uint8_t)(val & 0xFF)};
    write(this->fd, buf, 4);
}

uint8_t VL53L1X::read8Reg16(uint8_t addr, uint16_t reg) {
    ioctl(this->fd, I2C_SLAVE, addr);
    uint8_t reg_buf[2] = {(uint8_t)(reg >> 8), (uint8_t)(reg & 0xFF)};
    write(this->fd, reg_buf, 2);
    uint8_t val;
    read(this->fd, &val, 1);
    return val;
}

uint16_t VL53L1X::read16Reg16(uint8_t addr, uint16_t reg) {
    ioctl(this->fd, I2C_SLAVE, addr);
    uint8_t reg_buf[2] = {(uint8_t)(reg >> 8), (uint8_t)(reg & 0xFF)};
    write(this->fd, reg_buf, 2);
    uint8_t data[2];
    read(this->fd, data, 2);
    return (uint16_t)(data[0] << 8 | data[1]);
}

void VL53L1X::initialize() {
    for (uint8_t i = 0; i < 91; i++) {
        write8Reg16(this->address, 0x2D + i, DEFAULT_CONFIGURATION[i]);
    }
    this->startRanging();
    while (!this->isDataReady()) { std::this_thread::sleep_for(10ms); }
    this->clearInterrupt();
    this->stopRanging();
    write8Reg16(this->address, 0x002E, 0x09);
    write8Reg16(this->address, 0x002F, 0x00);
    this->interruptPolarity = !((read8Reg16(this->address, 0x0012) & 0x10) >> 4);
}

void VL53L1X::setAddress(uint8_t newAddress) {
    write8Reg16(this->address, 0x0001, newAddress & 0x7F);
    this->address = newAddress;
}

void VL53L1X::startRanging() { write8Reg16(this->address, 0x0087, 0x40); }
void VL53L1X::stopRanging() { write8Reg16(this->address, 0x0087, 0x00); }
void VL53L1X::clearInterrupt() { write8Reg16(this->address, 0x0086, 0x01); }
bool VL53L1X::isDataReady() { return (read8Reg16(this->address, 0x0031) & 0x01) == this->interruptPolarity; }

uint16_t VL53L1X::getDistance() {
    uint16_t d = read16Reg16(this->address, 0x0096);
    this->clearInterrupt();
    return d;
}