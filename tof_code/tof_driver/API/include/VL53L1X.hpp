#ifndef VL53L1X_HPP
#define VL53L1X_HPP

#include <cstdint>
#include <memory>

class VL53L1X {
public:
    using SharedPtr = std::shared_ptr<VL53L1X>;

    VL53L1X(int fd, uint8_t addr);
    
    static SharedPtr makeShared(int fd, uint8_t addr) {
        return std::make_shared<VL53L1X>(fd, addr);
    }

    void initialize();
    void startRanging();
    void stopRanging();
    void clearInterrupt();
    bool isDataReady();
    uint16_t getDistance();
    void setAddress(uint8_t newAddress);

    static const uint8_t DEFAULT_CONFIGURATION[91];

	// Helper functions for I2C
    void write8Reg16(uint8_t addr, uint16_t reg, uint8_t val);
    void write16Reg16(uint8_t addr, uint16_t reg, uint16_t val);
    uint8_t read8Reg16(uint8_t addr, uint16_t reg);
    uint16_t read16Reg16(uint8_t addr, uint16_t reg);

private:
    int fd;
    uint8_t address;
    uint8_t interruptPolarity;
};

#endif