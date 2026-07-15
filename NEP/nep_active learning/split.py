#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from ase.io import read, write

input_file = "to_add.xyz"

# 读取所有帧
atoms_list = read(input_file, index=":")

if len(atoms_list) == 0:
    raise ValueError(f"未从 {input_file} 读取到任何结构")

print(f"共读取到 {len(atoms_list)} 帧结构")

for i, atoms in enumerate(atoms_list, start=1):
    task_dir = f"task_{i:03d}"
    os.makedirs(task_dir, exist_ok=True)

    poscar_path = os.path.join(task_dir, "POSCAR")
    write(poscar_path, atoms, format="vasp")

    print(f"已写入: {poscar_path}")
print("完成")
