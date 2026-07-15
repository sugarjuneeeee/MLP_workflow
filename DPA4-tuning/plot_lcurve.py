#!/usr/bin/env python3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fname = "lcurve_finetune.out"

# ========== 安全读取正在写入的 lcurve 文件 ==========
with open(fname, "r") as f:
    lines = f.readlines()

header = None
rows = []

for line in lines:
    line = line.strip()
    if not line:
        continue

    if line.startswith("# step"):
        header = line.replace("#", "").split()
        continue

    if line.startswith("#"):
        continue

    parts = line.split()

    # 跳过训练正在写入导致的不完整行
    if header is not None and len(parts) != len(header):
        continue

    try:
        rows.append([float(x) for x in parts])
    except ValueError:
        continue

if header is None:
    raise RuntimeError("没有找到表头，请检查 lcurve_finetune.out")

if len(rows) == 0:
    raise RuntimeError("没有读到有效数据")

data = np.array(rows)

def col(name):
    if name not in header:
        raise RuntimeError(f"找不到列: {name}")
    return data[:, header.index(name)]

step = col("step")

# ========== 可选：滑动平均 ==========
def moving_average(y, window=20):
    if len(y) < window:
        return step, y
    kernel = np.ones(window) / window
    y_smooth = np.convolve(y, kernel, mode="valid")
    step_smooth = step[window - 1:]
    return step_smooth, y_smooth

use_smooth = True
smooth_window = 20

def plot_pair(ax, y_val_name, y_trn_name, ylabel, title):
    y_val = col(y_val_name)
    y_trn = col(y_trn_name)

    if use_smooth:
        sx_val, sy_val = moving_average(y_val, smooth_window)
        sx_trn, sy_trn = moving_average(y_trn, smooth_window)

        ax.plot(step, y_val, color="tab:blue", alpha=0.18)
        ax.plot(step, y_trn, color="tab:orange", alpha=0.18)

        ax.plot(sx_val, sy_val, color="tab:blue", lw=2, label=y_val_name)
        ax.plot(sx_trn, sy_trn, color="tab:orange", lw=2, label=y_trn_name)
    else:
        ax.plot(step, y_val, label=y_val_name)
        ax.plot(step, y_trn, label=y_trn_name)

    ax.set_yscale("log")
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(fontsize=9)

# ========== 合并作图 ==========
fig, axes = plt.subplots(3, 2, figsize=(13, 12))

# 1. Energy
plot_pair(
    axes[0, 0],
    "mae_e_val",
    "mae_e_trn",
    "Energy MAE",
    "Energy MAE"
)

# 2. Force
plot_pair(
    axes[0, 1],
    "mae_f_val",
    "mae_f_trn",
    "Force MAE / eV A$^{-1}$",
    "Force MAE"
)

# 3. Virial
plot_pair(
    axes[1, 0],
    "mae_v_val",
    "mae_v_trn",
    "Virial MAE",
    "Virial MAE"
)

# 4. Total RMSE
plot_pair(
    axes[1, 1],
    "rmse_val",
    "rmse_trn",
    "Total RMSE",
    "Total RMSE"
)

# 5. Learning rate
ax = axes[2, 0]
ax.plot(step, col("lr"), color="tab:green", lw=2, label="lr")
ax.set_yscale("log")
ax.set_xlabel("Step")
ax.set_ylabel("Learning rate")
ax.set_title("Learning Rate")
ax.grid(True, which="both", ls="--", alpha=0.3)
ax.legend(fontsize=9)

# 6. 最近 force MAE，线性坐标
ax = axes[2, 1]
recent_n = min(300, len(step))

ax.plot(
    step[-recent_n:],
    col("mae_f_val")[-recent_n:],
    color="tab:blue",
    lw=1.5,
    label="mae_f_val recent"
)
ax.plot(
    step[-recent_n:],
    col("mae_f_trn")[-recent_n:],
    color="tab:orange",
    lw=1.5,
    label="mae_f_trn recent"
)
ax.set_xlabel("Step")
ax.set_ylabel("Force MAE / eV A$^{-1}$")
ax.set_title(f"Recent Force MAE, last {recent_n} records")
ax.grid(True, ls="--", alpha=0.3)
ax.legend(fontsize=9)

fig.suptitle(
    f"Training curve summary, last step = {int(step[-1])}",
    fontsize=15
)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("lcurve_summary.png", dpi=300)
plt.close()

print("已生成: lcurve_summary.png")
print(f"当前最后 step: {int(step[-1])}")
print(f"mae_e_val = {col('mae_e_val')[-1]:.6e}")
print(f"mae_f_val = {col('mae_f_val')[-1]:.6e}")
print(f"mae_v_val = {col('mae_v_val')[-1]:.6e}")
print(f"rmse_val  = {col('rmse_val')[-1]:.6e}")
print(f"lr        = {col('lr')[-1]:.6e}")
