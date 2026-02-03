#!/bin/bash

#7 (OUT)
sudo busybox devmem 0x2448030 w 0xA

#15 (OUT)
sudo busybox devmem 0x2440020 w 0x5

#29 (IN)
sudo busybox devmem 0x2430068 w 0x58

#31 (OUT)
sudo busybox devmem 0x2430070 w 0x8

#32 (OUT)
sudo busybox devmem 0x2434080 w 0x5

#33 (OUT)
sudo busybox devmem 0x2434040 w 0x4