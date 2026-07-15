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
(nep_al) [gengzi@master SCF]$ ^C
(nep_al) [gengzi@master SCF]$ cat auto_submit.sh
#!/bin/bash

# ==========================================
# 🚀 Gen-3 专属：动态智能提交 (完美适配 vasp.sh)
# ==========================================

# 👉 核心设置区
MAX_JOBS=10                  # 队列中同时保持的任务数 (可根据超算限制自行微调)
SUBMIT_SCRIPT="vasp.sh"      # 你的提交脚本名称
USER_NAME=$(whoami)          # 自动获取你的账号名
TARGET_JOB_NAME="NEP_SCF"    # 🎯 核心追踪目标：严格匹配你 vasp.sh 里的 #SBATCH -J

echo "==================================================="
echo "🚀 启动动态提交流水线"
echo "🎯 目标追踪任务名：$TARGET_JOB_NAME，维持排队+运行数量：$MAX_JOBS"
echo "==================================================="

for task_dir in $(ls -d task_*/ | sort -V); do

    # 去掉末尾斜杠
    task_dir=${task_dir%/}

    # 尝试进入目录，进不去则跳过
    cd "$task_dir" || continue

    while true; do

        # 🌟 SLURM 原生高精度雷达：只统计名为 NEP_SCF 的任务
        current_jobs=$(squeue -u $USER_NAME --name="$TARGET_JOB_NAME" -h | wc -l)

        if [ "$current_jobs" -lt "$MAX_JOBS" ]; then
            break
        else
            echo "⏳ [$(date +'%H:%M:%S')] 专属队列满载 ($current_jobs/$MAX_JOBS)。休眠 60 秒... (等待提交: $task_dir)"
            sleep 60
        fi
    done

    echo "✅ [$(date +'%H:%M:%S')] 发现空位！正在将 $task_dir 注入超算队列..."
    sbatch "$SUBMIT_SCRIPT" > /dev/null 2>&1

    # 退回主目录
    cd ..

    # 停顿 2 秒缓冲
    sleep 2
done

echo "==================================================="
echo "🎉 恭喜！全部单点能任务均已成功流入调度管线！"
echo "==================================================="
