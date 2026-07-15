import numpy as np
from ase.io import read, write

def main():
    file_subset = 'train.xyz'        # 子集文件 
    file_full = 'NEP-dataset.xyz'    # 全集文件 
    file_output = 'train-active.xyz' # 输出结果

    print(f"[1/4] 正在读取子集文件 ({file_subset}) ...")
    frames_subset = read(file_subset, index=':')

    print(f"[2/4] 正在读取全集文件 ({file_full}) ...")
    frames_full = read(file_full, index=':')

    print("[3/4] 正在建立子集结构池 (准备执行浮点容差消耗匹配) ...")
    subset_pool = []
    for atoms in frames_subset:
        subset_pool.append({
            'n_atoms': len(atoms),
            'pos_sum': np.sum(atoms.positions), # 用于极速过滤的坐标和
            'cell': atoms.cell.array,
            'pos': atoms.positions,
            'numbers': atoms.numbers,
            'matched': False  # 这个标记非常重要！用于“消耗”机制
        })

    print("[4/4] 正在进行双重保险筛选 (浮点容差 + 一对一消耗) ...")
    diff_frames = []
    matched_count = 0

    # 设置容忍度：0.01（足够容忍各种文件读写带来的小数点截断误差）
    TOL = 1e-2
    total = len(frames_full)

    for i, f_atoms in enumerate(frames_full):
        if (i + 1) % 200 == 0:
            print(f"    处理进度: {i + 1} / {total} ...")

        n_f = len(f_atoms)
        pos_sum_f = np.sum(f_atoms.positions)
        pos_f = f_atoms.positions
        cell_f = f_atoms.cell.array
        num_f = f_atoms.numbers

        found_match = False

        for s in subset_pool:
            # 1. 如果该结构已经被消耗过了，直接跳过
            if s['matched']:
                continue

            # 2. 快速拦截：原子数必须相同
            if s['n_atoms'] != n_f:
                continue

            # 3. 快速拦截：坐标总和误差 (数学上极速拦截99%的无关构型)
            if abs(s['pos_sum'] - pos_sum_f) > (n_f * TOL):
                continue

            # 4. 严格检查：原子元素排列是否完全一致
            if not np.array_equal(s['numbers'], num_f):
                continue

            # 5. 严格检查：晶胞向量任何一个值的误差都必须 < 0.01
            if np.max(np.abs(s['cell'] - cell_f)) > TOL:
                continue

            # 6. 严格检查：所有原子的 XYZ 坐标误差都必须 < 0.01
            if np.max(np.abs(s['pos'] - pos_f)) > TOL:
                continue

            # 恭喜！完美匹配！标记为已消耗，不再参与后续匹配！
            s['matched'] = True
            found_match = True
            matched_count += 1
            break  # 跳出内层循环

        if not found_match:
            # 只有在子集池里找不到匹配（或者库存被消耗光了）时，才放入差集
            diff_frames.append(f_atoms)

    # ================== 打印统计报告 ==================
    print("\n" + "="*30 + " 统计报告 " + "="*30)
    print(f"全集构型数: {len(frames_full)}")
    print(f"子集构型数: {len(frames_subset)}")
    print(f"成功匹配并消耗: {matched_count}")
    print(f"差集构型数: {len(diff_frames)}")

    expected_count = len(frames_full) - len(frames_subset)
    if len(diff_frames) == expected_count:
        print("\n✨ 完美！运用 [容差匹配 + 消耗机制]，筛选出的数量与数学预期完全一致！")
    else:
        print(f"\n⚠️ 提示：筛出数量为 {len(diff_frames)}，预期为 {expected_count}。")
    print("="*70 + "\n")

    if diff_frames:
        write(file_output, diff_frames, format="extxyz")
        print(f"🎉 成功！差集结果已保存至: {file_output}")
    else:
        print("❌ 未检测到差异构型。")

if __name__ == "__main__":
    main()
