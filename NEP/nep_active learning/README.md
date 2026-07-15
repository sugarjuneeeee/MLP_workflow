在这里存放一些nep主动学习过程的脚本，基于官方的pynep代码库
workflow:
ps:tools.py asi_io.py maxvol.py maxvol_cpu.py maxcol_gpu.py为调用脚本，虽然不运行，但需要一起复制到工作目录中
1.由train.xyz训练出nep.txt
2.python select_active.py (in:nep.txt train.xyz out:active_set.asi)
3.run MD ,拼接MD轨迹为一个文件all_explore.lammpstrj，python merge.py
4.python reverse.py 为all_explore.lammpstrj添加假能量，力，维里标签，并转换为extxyz文件（in:all_explore.lammpstrj out:to_select.xyz）
5.python select_gamma.py 计算每一帧的外推gamma值,筛选符合要求结构（in:nep.txt to_select.xyz active_set.asi out:large_gamma.xyz）
6.python select_extend.py 将large_gamma.xyz中的结构和train.xyz中的结构合并计算maxvol,找出与train.xyz不重复的结构（in:nep.txt train.xyz large_gamma.xyz out:to_add.xyz）
7.python split.py 将to_add.xyz分解为独立的POSCAR，进行单点能计算(in:to_add.xyz out:task_***/POSCAR)
