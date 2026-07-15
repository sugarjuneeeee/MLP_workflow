已有预训练模型.pt和数据集extxyz
1.激活环境conda activate deepmd-dpa4
2.ase检查xyz python check_extxyz.py
3.将extendxyz转换为deepmd npy格式 python convert_finetune.py,生成dir deepmd_data(不同分子，组分，原子数要放在不同的子目录中)
4.准备微调输入文件finetune.json(model与预训练模型一致，type_map与预训练模型顺序一致)make_input_dpa4_air_finetune.py
5.微调命令：dp --pt train fintune.json --finetune 预训练模型.pt >finetune.log 2>&
6.冻结模型：dp --pt freeze -c model.ckpt.pt -o frozen_model.pth ——生成.pt2文件
7.测试模型：dp --pt test -m model.ckpt.pt -s deepmd_data/test  -d test.out
