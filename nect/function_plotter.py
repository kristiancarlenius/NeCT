import numpy as np
import matplotlib.pyplot as plt

def bowed_plateau(x, period=1.0, flat_ratio=0.75):
    """
    period     = length of one up-flat-down cycle
    flat_ratio = fraction of the cycle that is flat (0–1)
    """
    
    phase = (x % period) / period  # 0 → 1 inside each cycle
    
    rise_end = (1 - flat_ratio) / 2
    fall_start = 1 - rise_end
    
    y = np.zeros_like(phase)

    # Fast smooth rise
    rising = phase < rise_end
    y[rising] = np.sin((phase[rising] / rise_end) * (np.pi / 2))

    # Long flat top
    flat = (phase >= rise_end) & (phase <= fall_start)
    y[flat] = 1.0

    # Fast smooth fall
    falling = phase > fall_start
    y[falling] = np.sin(((1 - phase[falling]) / rise_end) * (np.pi / 2))

    return y


# Three full repeats
x = np.linspace(0, 6, 2000)
y = bowed_plateau(x, period=1.0, flat_ratio=0.6)

plt.figure(figsize=(7, 4))
plt.plot(x, y)

# Force only positive x and y
plt.xlim(0, 6)
plt.ylim(0, 1.1)

plt.axhline(0, linewidth=1)
plt.axvline(0, linewidth=1)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Three Repeating Bow Shapes with Long Flat Top")
plt.grid(True)
plt.show()
