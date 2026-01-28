## PART 1: ENVIRONMENT IDENTIFICATION

Each frame gets passed to the process_env function, but the model only runs every __buffer_size times, the rest of the time we returned the last cached environment. This is done to optimize the pipeline

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
