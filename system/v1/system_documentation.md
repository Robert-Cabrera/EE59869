## PART 1: INIT

- We initialize the models, load them all on the CUDA provider so that we can swap them at any time we need to. 
 
- We initialize the ToF sensors and its thread, this thread is constantly populating 3 different buffers and are locked until the main thread requires them via get_distances() to handle race conditions

## PART 2: SENSOR CHECKING AND PIPELINE OVERRIDE

- We check the distances from all 3 sensors: if any of them is below 0.9 (as of V1) we will fully stop and trigger a STOP branch

## PART 3: ENVIRONMENT IDENTIFICATION

Each frame gets passed to the process_env function, but the model only runs every __buffer_size times (10 as of V1), the rest of the time we returned the last cached environment. This is done to optimize the pipeline as the transformer CLIP model is insanely computationally expensive

### Sample Output:

```
[AI][ENV] Environment refreshed...
[AI] Frame 9: Indoors - Command: FORWARD
[AI] Frame 10: Indoors - Command: FORWARD
[AI] Frame 11: Indoors - Command: FORWARD
[AI] Frame 12: Indoors - Command: FORWARD
[AI] Frame 13: Indoors - Command: FORWARD
[AI] Frame 14: Indoors - Command: FORWARD
[AI] Frame 15: Indoors - Command: FORWARD
[AI] Frame 16: Indoors - Command: FORWARD
[AI] Frame 17: Indoors - Command: FORWARD
[AI] Frame 18: Indoors - Command: SLIGHT_RIGHT
[AI][ENV] Environment refreshed...
```
## PART 4: OUTDOOR AND INDOOR NAVIGATION

### Outdoors
- When the environment is detected as outdoor the system processes each frame through the ray casting + Bi-Seg navigation pipeline. This algorithm has been heavily optimized for the Jetson: the previous version ran on a python-loop to return the distances of each individual ray. For the embedded version we are vectorizing the calculating with NumPy and replaced cv2 countour-finding algorithm by a thresholded connected-component-finding algorithm. 

- After the decision on the frame is taken, we check if any crosswalks are intercepting with the bottom 50% of the frame, this will flag that a street has been detected and will be relayed to the user later on the pipeline.

- This pipeline thus returns the direction command AND the street_detected boolean.

### Indoors
- When the environment is detected as indoor we go through the MiDas + column scores pipeline, from here we simply return the direction command.
