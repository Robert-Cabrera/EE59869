from setuptools import setup, Extension

module = Extension('tof_driver',
                    sources=['tof_wrapper.cpp', 'API/src/VL53L1X.cpp', 'API/src/VL53L1X_default_config.cpp'],
                    include_dirs=['./API/include', '/usr/include'],
                    libraries=['gpiod'])

setup(name='ToFDriver',
      version='1.0',
      description='C++ ToF Driver for Python',
      ext_modules=[module])