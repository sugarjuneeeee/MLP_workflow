import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("test.s.out", comments="#")

dft = data[:, 0:9]
pred = data[:, 9:18]
err = pred - dft

names = [
    "xx", "xy", "xz",
    "yx", "yy", "yz",
    "zx", "zy", "zz"
]

fig, axes = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)

for i, ax in enumerate(axes.ravel()):
    ax.hist(err[:, i], bins=50, alpha=0.8)
    ax.axvline(0, color="r", ls="--")
    ax.set_title(f"stress {names[i]} error")
    ax.set_xlabel("Pred - DFT / eV/Å$^3$")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.3)

plt.savefig("stress_error_by_component.png", dpi=300)
