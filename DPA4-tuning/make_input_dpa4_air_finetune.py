import json
from copy import deepcopy

src = "DPA4-Air-OMat24-v20260627.json"
dst = "input_dpa4_air_finetune.json"

with open(src, "r") as f:
    j = json.load(f)

out = deepcopy(j)

# ============================================================
# 1. model 部分
# ============================================================
# 正式微调时，为了稳定，先关闭 compile
out["model"]["use_compile"] = False

# GPU 上 TF32 可以开；如果你想严格可复现，可以改成 False
out["model"]["enable_tf32"] = True


# ============================================================
# 2. learning rate
# ============================================================
# 对预训练 DPA4 微调，学习率不要太大
out["learning_rate"] = {
    "type": "cosine",
    "start_lr": 2.0e-6,
    "stop_lr": 1.0e-8,
    "warmup_ratio": 0.02,
    "warmup_start_factor": 0.2
}


# ============================================================
# 3. loss
# ============================================================
# 你的数据有 energy / force / virial

# 注意：
# virial 权重不建议一开始设太大，先用 0.1 比较稳。
# 如果 virial 拟合不够，再提高到 0.5 或 1.0。
out["loss"] = {
    "type": "ener",
    "loss_func": "mae",

    # 对不同原子数体系，建议保留
    "f_use_norm": True,

    # energy 权重
    "start_pref_e": 1.0,
    "limit_pref_e": 1.0,

    # force 权重，不额外强调
    "start_pref_f": 1.0,
    "limit_pref_f": 1.0,

    # virial 权重，有 virial 数据就不要设 0
    "start_pref_v": 0.1,
    "limit_pref_v": 0.1
}


# ============================================================
# 4. optimizer
# ============================================================
# 正式微调推荐 AdamW
out["optimizer"] = {
    "type": "AdamW",
    "weight_decay": 1.0e-6
}


# ============================================================
# 5. training data
# ============================================================
out["training"]["training_data"] = {
    "systems": "./data/train",
    "batch_size": "auto"
}

out["training"]["validation_data"] = {
    "systems": "./data/valid",
    "batch_size": "auto"
}

# 正式训练不要只验证 1 个 batch
# 如果原模板里有 numb_batch，这里删掉，让程序按默认方式验证
out["training"]["validation_data"].pop("numb_batch", None)


# ============================================================
# 6. training settings
# ============================================================
# 正式微调轮数
out["training"]["num_epochs"] = 100

# 保存 checkpoint
out["training"]["save_freq"] = 1000
out["training"]["max_ckpt_keep"] = 5

# 日志
out["training"]["disp_file"] = "lcurve_finetune.out"
out["training"]["disp_freq"] = 100
out["training"]["disp_avg"] = True
out["training"]["disp_training"] = True
out["training"]["time_training"] = True

# 正式微调建议开启 EMA
out["training"]["enable_ema"] = True
out["training"]["ema_decay"] = 0.999
out["training"]["ema_ckpt_keep"] = 3

# 不开 tensorboard/profiler，保持简单
out["training"]["tensorboard"] = False
out["training"]["enable_profiler"] = False
out["training"]["profiling"] = False

# 单卡一般用 0
out["training"]["zero_stage"] = 0

# 梯度裁剪
out["training"]["gradient_max_norm"] = 5.0

# 本地微调数据一般不需要官方 stat_file
out["training"].pop("stat_file", None)


with open(dst, "w") as f:
    json.dump(out, f, indent=4)

print(f"written: {dst}")
