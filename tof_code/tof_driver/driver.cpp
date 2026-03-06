#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <stdint.h>
#include <gpiod.h>
#include <iostream>
#include <thread>
#include <chrono>
#include "API/include/VL53L1X.hpp"

#define P_OUT 1
#define SLEEP_MS(ms) std::this_thread::sleep_for(std::chrono::milliseconds(ms))

struct gpiod_chip *CHIP;
struct gpiod_line *RIGHT_T, *CENTER_T, *LEFT_T; 

void locate_pins(int *right_pin, int *center_pin, int* left_pin) {
    FILE *fp;
    char buffer[256];
    const char *cmd = "echo \"seniordesign\" | sudo -S gpioinfo | grep -E \"PH.00|PQ.06|PG.06\"";
    fp = popen(cmd, "r");
    if (fp == NULL) return;
    
    while (fgets(buffer, sizeof(buffer), fp) != NULL){
        int offset;
        char name[32];
        if (sscanf(buffer, " line %d: \"%[^\"]\"", &offset, name) == 2) {
            if (strcmp(name, "PH.00") == 0) *right_pin = offset;
            else if (strcmp(name, "PQ.06") == 0) *center_pin = offset;
            else if (strcmp(name, "PG.06") == 0) *left_pin = offset;
        }
    }
    pclose(fp);
    printf("Parsed Pins -> Right: %d, Center: %d, Left: %d\n", *right_pin, *center_pin, *left_pin);
}

int gpio_init() {
    CHIP = gpiod_chip_open_by_name("gpiochip0"); 
    return (CHIP != NULL);
}

struct gpiod_line* gpio_mode(int offset, int mode) {
    struct gpiod_line *line = gpiod_chip_get_line(CHIP, offset);
    if (!line) return NULL;
    if (mode) gpiod_line_request_output(line, "orin_out", 0);
    else gpiod_line_request_input(line, "orin_in");
    return line;
}

void gpio_write(struct gpiod_line *line, int value) {
    if (line) gpiod_line_set_value(line, value);
}
int main() {
    if (!gpio_init()) {
        fprintf(stderr, "Failed to open gpiochip\n");
        return -1;
    }

    int r_offset, c_offset, l_offset;
    locate_pins(&r_offset, &c_offset, &l_offset);

    RIGHT_T  = gpio_mode(r_offset, P_OUT);
    CENTER_T = gpio_mode(c_offset, P_OUT);
    LEFT_T   = gpio_mode(l_offset, P_OUT);

    // 1. HARD RESET: Kill power to all sensors and wait
    printf("Resetting all sensors...\n");
    gpio_write(RIGHT_T, 0);
    gpio_write(CENTER_T, 0);
    gpio_write(LEFT_T, 0);
    SLEEP_MS(500); // Increased to ensure capacitors discharge

    int i2c_fd = open("/dev/i2c-7", O_RDWR);
    if (i2c_fd < 0) {
        perror("Failed to open I2C bus 7");
        return -1;
    }

    // Create sensor objects pointing to default 0x29
    auto sensorR = VL53L1X::makeShared(i2c_fd, 0x29);
    auto sensorC = VL53L1X::makeShared(i2c_fd, 0x29);
    auto sensorL = VL53L1X::makeShared(i2c_fd, 0x29);

    printf("Starting sequential initialization...\n");

    // --- Initialize RIGHT (Using extra patience) ---
    gpio_write(RIGHT_T, 1);
    SLEEP_MS(150); 
    
    // Verify it's actually awake at 0x29 before changing address
    uint16_t idR = sensorR->read16Reg16(0x29, 0x010F);
    if (idR != 0xEACC) printf("Warning: Right sensor not responding at default 0x29!\n");
    
    sensorR->setAddress(0x30);
    SLEEP_MS(50);
    sensorR->initialize();
    sensorR->startRanging();
    printf("Right sensor initialized at 0x30\n");

    // --- Initialize CENTER ---
    gpio_write(CENTER_T, 1);
    SLEEP_MS(150);
    sensorC->setAddress(0x32);
    SLEEP_MS(50);
    sensorC->initialize();
    sensorC->startRanging();
    printf("Center sensor initialized at 0x32\n");

    // --- Initialize LEFT ---
    gpio_write(LEFT_T, 1);
    SLEEP_MS(150);
    sensorL->setAddress(0x34);
    SLEEP_MS(50);
    sensorL->initialize();
    sensorL->startRanging();
    printf("Left sensor initialized at 0x34\n");

    printf("\nReading Distances (mm):\n");
    printf("--------------------------------------------------\n");

    while (true) {
        // We use a small check to ensure we only print valid data
        uint16_t dR = sensorR->getDistance();
        uint16_t dC = sensorC->getDistance();
        uint16_t dL = sensorL->getDistance();

        printf("Right: %4u | Center: %4u | Left: %4u   \r", dR, dC, dL);
        fflush(stdout);
        
        // Slightly slower loop to prevent I2C bus congestion
        SLEEP_MS(60); 
    }

    close(i2c_fd);
    gpiod_chip_close(CHIP);
    return 0;
}