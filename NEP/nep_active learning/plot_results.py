import numpy as np
import matplotlib.pyplot as plt
import os

print("🚀 启动 (终极全能版) NEP 画图脚本: 正在聚合能量、受力、维里应力...")

# 设置1x3的画布，尺寸为 18x6，适合宽屏查看或放入论文补充材料
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 统一的画图函数
def plot_parity(ax, filename, label, units):
    if not os.path.exists(filename):
        ax.text(0.5, 0.5, f"未找到 {filename}", ha='center', va='center', fontsize=15, color='red')
        ax.set_title(label, fontsize=16)
        return

    # 读取数据
    data = np.loadtxt(filename)
    
    # 💡 极其聪明的自动拆分逻辑：
    # Energy 有 2 列 (1对1)
    # Force 有 6 列 (3对3)
    # Virial 有 12 列 (6对6)
    half_cols = data.shape[1] // 2
    real_val = data[:, :half_cols].flatten()
    pred_val = data[:, half_cols:].flatten()

    # 计算 RMSE
    rmse = np.sqrt(np.mean((real_val - pred_val)**2))
    
    # 画散点，调低 alpha(透明度) 让重叠的点显示出密度（颜色加深）
    ax.scatter(real_val, pred_val, s=2, alpha=0.4, color='#1f77b4')
    
    # 获取数据的边界，画完美的对角线
    min_val = min(np.min(real_val), np.min(pred_val))
    max_val = max(np.max(real_val), np.max(pred_val))
    
    # 给坐标轴留 5% 的余量，图更好看
    margin = (max_val - min_val) * 0.05
    if margin == 0: margin = 0.1
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)
    
    # 画 y=x 虚线
    ax.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin], 'k--', lw=2)

    # 美化图表
    ax.set_xlabel(f"DFT {label} ({units})", fontsize=14)
    ax.set_ylabel(f"NEP {label} ({units})", fontsize=14)
    ax.set_title(f"{label} Correlation\nRMSE = {rmse:.4f} {units}", fontsize=16)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # 强制让 X 轴和 Y 轴比例 1:1，这是散点对角线图的灵魂！
    ax.set_aspect('equal', adjustable='box')
    
    print(f"📊 {label:<8} RMSE = {rmse:.4f} {units}")

# 分别把三个物理量画在三个子图上
plot_parity(axes[0], "energy_train.out", "Energy", "eV/atom")
plot_parity(axes[1], "force_train.out", "Force", "eV/Å")
plot_parity(axes[2], "virial_train.out", "Virial", "eV/atom")

# 自动调整间距并保存
plt.tight_layout()
plt.savefig("train_results_all.png", dpi=300, bbox_inches='tight')
print("✅ 画图成功！快去打开 train_results_all.png 看看你的模型表现吧！")
