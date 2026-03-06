#include <Python.h>
#include "VL53L1X.hpp"
#include <gpiod.h>
#include <fcntl.h>
#include <unistd.h>

// Global pointers for the sensors
std::shared_ptr<VL53L1X> sensorR, sensorC, sensorL;
struct gpiod_chip *chip;
struct gpiod_line *lineR, *lineC, *lineL;

// Utility to sleep within C++
void sleep_ms(int ms) {
    usleep(ms * 1000);
}

static PyObject* py_init_tof(PyObject* self, PyObject* args) {
    chip = gpiod_chip_open_by_name("gpiochip0");
    // Using your parsed offsets (adjust if these change)
    lineR = gpiod_chip_get_line(chip, 43); 
    lineC = gpiod_chip_get_line(chip, 106);
    lineL = gpiod_chip_get_line(chip, 41);

    gpiod_line_request_output(lineR, "tof", 0);
    gpiod_line_request_output(lineC, "tof", 0);
    gpiod_line_request_output(lineL, "tof", 0);

    // Reset Sequence
    gpiod_line_set_value(lineR, 0); gpiod_line_set_value(lineC, 0); gpiod_line_set_value(lineL, 0);
    sleep_ms(300);

    int fd = open("/dev/i2c-7", O_RDWR);
    sensorR = VL53L1X::makeShared(fd, 0x29);
    sensorC = VL53L1X::makeShared(fd, 0x29);
    sensorL = VL53L1X::makeShared(fd, 0x29);

    // Sequential Init logic that worked
    gpiod_line_set_value(lineR, 1); sleep_ms(150);
    sensorR->setAddress(0x30); sleep_ms(50);
    sensorR->initialize(); sensorR->startRanging();

    gpiod_line_set_value(lineC, 1); sleep_ms(150);
    sensorC->setAddress(0x32); sleep_ms(50);
    sensorC->initialize(); sensorC->startRanging();

    gpiod_line_set_value(lineL, 1); sleep_ms(150);
    sensorL->setAddress(0x34); sleep_ms(50);
    sensorL->initialize(); sensorL->startRanging();

    Py_RETURN_NONE;
}

static PyObject* py_get_distances(PyObject* self, PyObject* args) {
    // Return a dictionary matching your Python code's expectation
    return Py_BuildValue("{s:i, s:i, s:i}", 
        "right",  (int)sensorR->getDistance(),
        "center", (int)sensorC->getDistance(),
        "left",   (int)sensorL->getDistance());
}

static PyObject* py_cleanup_tof(PyObject* self, PyObject* args) {
    gpiod_chip_close(chip);
    Py_RETURN_NONE;
}

// Module definition
static PyMethodDef ToFMethods[] = {
    {"init_tof", py_init_tof, METH_VARARGS, "Init ToF"},
    {"get_distances", py_get_distances, METH_VARARGS, "Read ToF"},
    {"cleanup_tof", py_cleanup_tof, METH_VARARGS, "Cleanup ToF"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef tofmodule = { PyModuleDef_HEAD_INIT, "tof_driver", NULL, -1, ToFMethods };

PyMODINIT_FUNC PyInit_tof_driver(void) { return PyModule_Create(&tofmodule); }