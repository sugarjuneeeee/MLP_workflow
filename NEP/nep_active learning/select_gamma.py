import numpy as np
from ase.io import write
from pynep.io import load_nep, dump_nep
from tools import get_gamma

print("启动双端截断 Gamma 筛选器...")

nep_file = "nep.txt"             #指定所使用的nep势
traj = load_nep("to_select.xyz") #指定计算gamma值的数据集

print("⏳ 正在调用 pynep 计算所有帧的 Gamma 外推等级...")
get_gamma(traj, nep_file, "active_set.asi")

GAMMA_MIN = 3.0     #gamma值下限
GAMMA_MAX = 50.0    #gamma值上限

out_traj = []
count_low = 0
count_high = 0

for atoms in traj:
    max_g = atoms.arrays["gamma"].max()

    if max_g < GAMMA_MIN:
        count_low += 1
    elif max_g > GAMMA_MAX:
        count_high += 1
    else:
        out_traj.append(atoms)

print(f"   总输入帧数: {len(traj)}")
print(f"   简单结构(Gamma < {GAMMA_MIN}): {count_low} 帧")
print(f"   防崩溃拦截 (Gamma > {GAMMA_MAX}): {count_high} 帧 ")
print(f"   有效结构: {len(out_traj)} 帧")
print("-" * 50)

# 输出
if len(out_traj) > 0:
    try:
        dump_nep("large_gamma.xyz", out_traj)
    except:
        write("large_gamma.xyz", out_traj)
    print(f"💾 已将 {len(out_traj)} 个极品结构保存至 large_gamma.xyz")
else:
    print("⚠️ 警告：当前截断范围内没有挑出任何结构！尝试放宽 GAMMA_MIN 或修改 MD 探索条件。")
