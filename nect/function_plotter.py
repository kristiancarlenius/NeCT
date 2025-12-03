import numpy as np
import matplotlib.pyplot as plt

def bowed_plateau(x, period=1.0, flat_ratio=0.6):
    """
    Smooth bow: fast rise → flat → fast fall, repeated.
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


def square_pulse(x, period=1.0, pulse_ratio=0.15):
    """
    Square/rectangular function:
    - 0 most of the period
    - 1 for a short central pulse (much shorter width than bowed plateau).
    """
    phase = (x % period) / period  # 0 → 1 inside each cycle

    # Centered pulse around phase = 0.5
    half_pulse = pulse_ratio / 2
    y = np.zeros_like(phase)

    on_region = (phase >= 0.5 - half_pulse) & (phase <= 0.5 + half_pulse)
    y[on_region] = 1.0

    return y


# Three full repeats over x ∈ [0, 3]
x = np.linspace(0, 6, 2000)

y_bow   = bowed_plateau(x, period=1.0, flat_ratio=0.6)
y_square = square_pulse(x, period=0.5, pulse_ratio=0.15)  # much shorter width

plt.figure(figsize=(7, 4))
plt.plot(x, y_bow,    label="Bow with long flat top")
plt.plot(x, y_square, label="Short square pulse")

# Only positive x and y
plt.xlim(0, 6)
plt.ylim(0, 1.1)

plt.axhline(0, linewidth=1)
plt.axvline(0, linewidth=1)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Bow-Shaped Function and Narrow Square Function (3 repeats)")
plt.grid(True)
plt.legend()
plt.show()
