#!/bin/bash

# ===== 基础输入文件检查 =====
required_files=("INCAR" "vasp.sh" "POTCAR")

echo "检查当前目录下必要文件..."

for f in "${required_files[@]}"; do
    if [ ! -f "$f" ]; then
        echo "错误：当前目录下缺少 $f"
        exit 1
    fi
done

echo "必要文件齐全"
echo

# ===== 统计信息 =====
total_tasks=0
ok_tasks=0
skip_tasks=0

missing_poscar_dirs=()

# ===== 遍历 task 子目录 =====
for dir in task_*; do
    [ -d "$dir" ] || continue

    total_tasks=$((total_tasks + 1))

    echo "处理目录: $dir"

    # 检查 POSCAR
    if [ ! -f "$dir/POSCAR" ]; then
        echo "  跳过：缺少 POSCAR"
        missing_poscar_dirs+=("$dir")
        skip_tasks=$((skip_tasks + 1))
        echo
        continue
    fi

    # 复制 INCAR 和 vasp.sh
    cp INCAR "$dir/" && echo "  已复制 INCAR"
    cp vasp.sh "$dir/" && echo "  已复制 vasp.sh"

    # 创建/更新 POTCAR 软链接
    ln -sfn ../POTCAR "$dir/POTCAR"
    echo "  已链接 POTCAR -> ../POTCAR"

    ok_tasks=$((ok_tasks + 1))
    echo
done

# ===== 输出总结 =====
echo "===================="
echo "处理完成"
echo "总 task 目录数: $total_tasks"
echo "成功处理数:    $ok_tasks"
echo "跳过数:        $skip_tasks"

if [ ${#missing_poscar_dirs[@]} -gt 0 ]; then
    echo
    echo "以下目录缺少 POSCAR，未处理："
    for d in "${missing_poscar_dirs[@]}"; do
        echo "  $d"
    done
fi
