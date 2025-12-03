import numpy as np
import matplotlib.pyplot as plt

# Bow-shaped function that repeats every `period`
def bow_function(x, period=1.0):
    # Where are we inside the current period? (0 to 1)
    phase = (x % period) / period
    # Smooth "bow" going up and down, touching 0 at the ends
    y = np.sin(np.pi * phase)  # sin(0)=0, sin(pi/2)=1, sin(pi)=0
    return y  # already non-negative on [0,1]

# Make x from 0 to 3 (three bows with period 1)
x = np.linspace(0, 3, 1000)
y = bow_function(x, period=1.0)

plt.figure(figsize=(6, 4))
plt.plot(x, y)

# Only positive x and y axes (first quadrant)
plt.xlim(0, 3)
plt.ylim(0, 1.1)

# Draw the axes at x=0 and y=0
plt.axhline(0, linewidth=1)
plt.axvline(0, linewidth=1)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Repeating Bow-Shaped Function (3 times)")
plt.grid(True)
plt.show()
