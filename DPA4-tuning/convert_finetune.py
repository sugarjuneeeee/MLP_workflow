import os
import re
import random
import shutil
import numpy as np
from ase.io import read
import dpdata


# ===================== 配置区 =====================
xyz_filename = "train.xyz"
output_dir = "deepmd_data"

train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1

random_seed = 42

# 固定元素顺序，必须和 DeepMD input.json 里的 type_map 一致
# 对微调尤其重要：必须和预训练模型的 type_map 顺序一致
atom_names_fixed = ["O", "Sr", "Ti"]


# ===================== 工具函数 =====================
def clean_output_dir(path):
    if os.path.exists(path):
        print(f"🧹 删除旧目录: {path}")
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def extract_energies_from_extxyz(filename):
    energies = []
    energy_pattern = re.compile(r'\b[Ee]nergy=([-+0-9.eE]+)')

    with open(filename, "r") as f:
        for line in f:
            match = energy_pattern.search(line)
            if match:
                energies.append(float(match.group(1)))

    return energies


def extract_forces(atoms, frame_index):
    # 1. 优先从 atoms.arrays 读取
    for key in ["forces", "force", "Forces", "Force"]:
        if key in atoms.arrays:
            forces = np.asarray(atoms.arrays[key], dtype=float)
            return forces

    # 2. 模糊匹配 force 字段
    for key in atoms.arrays.keys():
        if "force" in key.lower():
            forces = np.asarray(atoms.arrays[key], dtype=float)
            return forces

    # 3. ASE 标准接口
    try:
        forces = atoms.get_forces()
        if forces is not None:
            forces = np.asarray(forces, dtype=float)
            return forces
    except Exception:
        pass

    # 4. 从 atoms.calc.results 读取
    if atoms.calc is not None:
        for key in ["forces", "force", "Forces", "Force"]:
            if key in atoms.calc.results:
                forces = np.asarray(atoms.calc.results[key], dtype=float)
                return forces

        for key in atoms.calc.results.keys():
            if "force" in key.lower():
                forces = np.asarray(atoms.calc.results[key], dtype=float)
                return forces

    calc_keys = list(atoms.calc.results.keys()) if atoms.calc is not None else None

    raise ValueError(
        f"第 {frame_index} 帧没有找到 forces 数据。\n"
        f"atoms.arrays keys = {list(atoms.arrays.keys())}\n"
        f"atoms.info keys = {list(atoms.info.keys())}\n"
        f"atoms.calc.results keys = {calc_keys}"
    )


def extract_virial(atoms):
    # 常见情况：virial 或 stress 在 atoms.info 里
    for key in ["virial", "Virial", "stress", "Stress"]:
        if key in atoms.info:
            value = atoms.info[key]

            if isinstance(value, str):
                value = [
                    float(x)
                    for x in value.replace("[", " ").replace("]", " ").split()
                ]

            value = np.asarray(value, dtype=float)

            if value.size == 9:
                return value.reshape(3, 3)

            if value.size == 6:
                # ASE stress 常见 6 分量顺序: xx yy zz yz xz xy
                xx, yy, zz, yz, xz, xy = value
                return np.array(
                    [
                        [xx, xy, xz],
                        [xy, yy, yz],
                        [xz, yz, zz],
                    ],
                    dtype=float,
                )

            raise ValueError(f"{key} 数据长度不是 9 或 6，而是 {value.size}")

    # 从 calc.results 读取
    if atoms.calc is not None:
        for key in ["virial", "Virial", "stress", "Stress"]:
            if key in atoms.calc.results:
                value = np.asarray(atoms.calc.results[key], dtype=float)

                if value.size == 9:
                    return value.reshape(3, 3)

                if value.size == 6:
                    xx, yy, zz, yz, xz, xy = value
                    return np.array(
                        [
                            [xx, xy, xz],
                            [xy, yy, yz],
                            [xz, yz, zz],
                        ],
                        dtype=float,
                    )

                raise ValueError(f"{key} 数据长度不是 9 或 6，而是 {value.size}")

    return None


def build_labeled_system(atoms, energy, frame_index):
    symbols = atoms.get_chemical_symbols()

    unknown_elements = sorted(set(symbols) - set(atom_names_fixed))
    if unknown_elements:
        raise ValueError(
            f"第 {frame_index} 帧发现未声明元素: {unknown_elements}\n"
            f"当前 atom_names_fixed = {atom_names_fixed}"
        )

    atom_numbs = [symbols.count(name) for name in atom_names_fixed]
    atom_types = [atom_names_fixed.index(symbol) for symbol in symbols]

    forces = extract_forces(atoms, frame_index)

    if forces.shape != (len(symbols), 3):
        raise ValueError(
            f"第 {frame_index} 帧 forces 形状错误: {forces.shape}, "
            f"应为 ({len(symbols)}, 3)"
        )

    virial = extract_virial(atoms)

    data_dict = {
        "atom_names": atom_names_fixed,
        "atom_numbs": atom_numbs,
        "atom_types": np.asarray(atom_types, dtype=int),
        "coords": np.asarray([atoms.get_positions()], dtype=float),
        "cells": np.asarray([atoms.cell.array], dtype=float),
        "energies": np.asarray([energy], dtype=float),
        "forces": np.asarray([forces], dtype=float),
        "orig": np.asarray([0.0, 0.0, 0.0], dtype=float),
    }

    if virial is not None:
        data_dict["virials"] = np.asarray([virial], dtype=float)

    system = dpdata.LabeledSystem()
    system.data = data_dict

    return system


# ===================== 主流程 =====================
def main():
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-8:
        raise ValueError(
            f"train_ratio + valid_ratio + test_ratio 必须等于 1，"
            f"当前为 {train_ratio + valid_ratio + test_ratio}"
        )

    print(f"⏳ 读取 extxyz 文件: {xyz_filename}")
    frames = read(xyz_filename, index=":")
    n_frames = len(frames)
    print(f"📊 读取到 {n_frames} 帧结构。")

    if n_frames == 0:
        raise ValueError("没有读取到任何结构。")

    print("⚡ 从 extxyz 注释行提取 Energy/energy ...")
    raw_energies = extract_energies_from_extxyz(xyz_filename)

    if len(raw_energies) != n_frames:
        raise ValueError(
            f"能量数量与结构数量不一致: energies={len(raw_energies)}, "
            f"frames={n_frames}"
        )

    print(f"✅ 成功提取 {len(raw_energies)} 个能量。")

    clean_output_dir(output_dir)

    train_ms = dpdata.MultiSystems()
    valid_ms = dpdata.MultiSystems()
    test_ms = dpdata.MultiSystems()

    random.seed(random_seed)

    indices = list(range(n_frames))
    random.shuffle(indices)

    n_train = int(n_frames * train_ratio)
    n_valid = int(n_frames * valid_ratio)

    train_indices = indices[:n_train]
    valid_indices = indices[n_train:n_train + n_valid]
    test_indices = indices[n_train + n_valid:]

    if len(train_indices) == 0:
        raise ValueError("训练集为空，请增大 train_ratio 或增加数据。")

    if len(valid_indices) == 0:
        raise ValueError("验证集为空，请增大 valid_ratio 或增加数据。")

    if len(test_indices) == 0:
        print("⚠️ 测试集为空。如果你需要最终独立测试，请增大 test_ratio 或增加数据。")

    print("⏳ 转换训练集 ...")
    for frame_index in train_indices:
        system = build_labeled_system(
            atoms=frames[frame_index],
            energy=raw_energies[frame_index],
            frame_index=frame_index,
        )
        train_ms.append(system)

    print("⏳ 转换验证集 ...")
    for frame_index in valid_indices:
        system = build_labeled_system(
            atoms=frames[frame_index],
            energy=raw_energies[frame_index],
            frame_index=frame_index,
        )
        valid_ms.append(system)

    print("⏳ 转换测试集 ...")
    for frame_index in test_indices:
        system = build_labeled_system(
            atoms=frames[frame_index],
            energy=raw_energies[frame_index],
            frame_index=frame_index,
        )
        test_ms.append(system)

    train_dir = os.path.join(output_dir, "training_data")
    valid_dir = os.path.join(output_dir, "validation_data")
    test_dir = os.path.join(output_dir, "test_data")

    print(f"💾 保存训练集: {train_dir}")
    train_ms.to("deepmd/npy", train_dir)

    print(f"💾 保存验证集: {valid_dir}")
    valid_ms.to("deepmd/npy", valid_dir)

    if len(test_indices) > 0:
        print(f"💾 保存测试集: {test_dir}")
        test_ms.to("deepmd/npy", test_dir)

    print("\n🎉 转换完成。")
    print(f"训练集帧数: {len(train_indices)}")
    print(f"验证集帧数: {len(valid_indices)}")
    print(f"测试集帧数: {len(test_indices)}")
    print(f"元素顺序 atom_names: {atom_names_fixed}")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
