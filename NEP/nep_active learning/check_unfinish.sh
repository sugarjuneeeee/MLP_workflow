#!/bin/bash

echo "正在检查 VASP 任务计算状态..."
echo "-----------------------------------"

# 初始化数组来存储已完成和未完成的任务
finished_tasks=()
unfinished_tasks=()
not_started=()

# 遍历所有 task_ 开头的目录
for dir in task_*; do
    if [ -d "$dir" ]; then
        if [ -f "$dir/OUTCAR" ]; then
            # 检查 OUTCAR 中是否有 VASP 正常结束的标志
            if tail -n 50 "$dir/OUTCAR" | grep -q "General timing and accounting"; then
                finished_tasks+=("$dir")
            else
                unfinished_tasks+=("$dir")
            fi
        else
            not_started+=("$dir")
        fi
    fi
done

# 打印结果
echo -e "\033[32m=== 已完成的任务 (${#finished_tasks[@]}) ===\033[0m"
for t in "${finished_tasks[@]}"; do
    echo -n "$t  "
done
echo -e "\n"

echo -e "\033[33m=== 正在计算或中断的任务 (${#unfinished_tasks[@]}) ===\033[0m"
for t in "${unfinished_tasks[@]}"; do
    echo -n "$t  "
done
echo -e "\n"

if [ ${#not_started[@]} -gt 0 ]; then
    echo -e "\033[31m=== 未找到 OUTCAR (尚未开始) 的任务 (${#not_started[@]}) ===\033[0m"
    for t in "${not_started[@]}"; do
        echo -n "$t  "
    done
    echo -e "\n"
fi

echo "-----------------------------------"
echo "检查完毕。"
