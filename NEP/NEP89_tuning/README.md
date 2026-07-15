这里存放一些微调NEP89时所需代码
如果已有数据集：
1.对现有数据集运行python select_active.py
2.使用生成的active_set.xyz进行预测
3.将数据集减去active_set.xyz，剩余的结构可以划分为测试集和预测集
