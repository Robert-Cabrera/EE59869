# Context
We have an array of 3 TOF Sensors, they are connected to 4 common columns:

-SDA on PIN3
-SCL on PIN5
-VDD on the 3.3V PIN
-VSS on the GND PIN

There's then 3 individual pins (one for each sensor) connected from the xSHUT pins to some GPIO pins

-ToF (left) on PIN32
-ToF (center) on PIN31
-ToF (right) on PIN33

# How to check wiring
Note that the I2C pins (SDA and SCL) are connected to the __I2C1__ peripheral. In order to see if the sensors are being recognized you need to run:

```shell
sudo i2cdetect -y -r 7
```

Also note that the xSHUT pins need to be driven high, and the GPIO pins are active low. In software we will toggle them on, but if you are conducting individual testing and you are not getting signal response, try driving the xSHUT high or disconnecting it.

# Sample Output

```
[STATUS = GOOD] left:38.8
[STATUS = GOOD] center:42.0
[STATUS = GOOD] right:46.8

[STATUS = GOOD] left:38.3
[STATUS = GOOD] center:41.9
[STATUS = GOOD] right:45.8

[STATUS = GOOD] left:38.3
[STATUS = GOOD] center:41.5
[STATUS = GOOD] right:46.9

[STATUS = GOOD] left:36.8
[STATUS = GOOD] center:42.0
[STATUS = GOOD] right:45.4

[STATUS = GOOD] left:38.1
[STATUS = GOOD] center:41.6
[STATUS = GOOD] right:44.4

[STATUS = GOOD] left:37.7
[STATUS = GOOD] center:41.7
[STATUS = GOOD] right:44.8
```