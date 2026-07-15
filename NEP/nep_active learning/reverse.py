import os
import numpy as np
from ase.io import read, write
from ase import Atoms

print("🚀 开始智能解析并转换多尺度 LAMMPS 轨迹...")

# 1. 明确告诉脚本 LAMMPS 里的 Type 对应什么真实元素
# 这必须和你 in 文件里的 pair_coeff * * nep.txt Sr Ti O 严格对应！
type_map = {
    1: 'Sr', 
    2: 'Ti', 
    3: 'O'
}

# 2. 读取合并好的超级轨迹 (或者单个 dump 文件)
input_traj = 'all_explore.lammpstrj' # 根据你实际的文件名修改
try:
    traj_raw = read(input_traj, index=':', format='lammps-dump-text')
    print(f"✅ 成功读取轨迹文件，共包含 {len(traj_raw)} 帧结构。")
except Exception as e:
    print(f"❌ 读取失败，请检查文件是否存在！报错: {e}")
    exit()

traj_clean = []

# 3. 动态重构每一帧 (核心魔法升级版)
for i, atoms in enumerate(traj_raw):
    # ASE 读取 LAMMPS dump 时，默认会把 Type 1, 2, 3 存成原子序数 (1=H, 2=He, 3=Li)
    # 我们把这些代号提取出来，通过 type_map 动态翻译成真实的 Sr, Ti, O
    raw_types = atoms.numbers 
    
    try:
        real_symbols = [type_map[t] for t in raw_types]
    except KeyError as e:
        print(f"❌ 严重错误: 在第 {i} 帧发现了未知的原子类型 {e}！请检查 dump 文件。")
        exit()

    # 捏造纯净晶胞：自动适应当前的原子数和盒子大小
    clean_atoms = Atoms(
        symbols=real_symbols,
        positions=atoms.positions,
        cell=atoms.cell,
        pbc=True
    )
    
    # 继续伪造 pynep 死活非要的能量、应力和受力标签
    clean_atoms.info['energy'] = 0.0
    clean_atoms.info['virial'] = np.zeros((3, 3))
    clean_atoms.arrays['force'] = np.zeros((len(clean_atoms), 3))

    traj_clean.append(clean_atoms)

# 4. 输出最终的合法 extxyz 格式
output_name = 'to_select.xyz'
write(output_name, traj_clean, format='extxyz')

print("\n===================================================")
print(f"🎉 智能转换彻底完成！已保存为: {output_name}")
print(f"📏 完美适配了不同尺寸的超胞和各类缺陷结构。")
print("===================================================")
