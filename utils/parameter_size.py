# # 定义总参数量、可训练参数量及非可训练参数量变量
# Total_params = 0
# Trainable_params = 0
# NonTrainable_params = 0
#
# # 遍历model.parameters()返回的全局参数列表
# for param in model.parameters():
#     mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
#     Total_params += mulValue  # 总参数量
#     if param.requires_grad:
#         Trainable_params += mulValue  # 可训练参数量
#     else:
#         NonTrainable_params += mulValue  # 非可训练参数量
#
# print(f'Total params: {Total_params}')
# print(f'Trainable params: {Trainable_params}')
# print(f'Non-trainable params: {NonTrainable_params}')

import torch
import dnnlib

# 加载模型文件
model_weights = "/data/HDD1/tjut_liuao/FU-GUIDE/experiments3/guide/checkpoints/last.pkl"

with dnnlib.util.open_url(model_weights) as f:
    # 加载预训练的生成器模型
    init_global_generator = legacy.load_network_pkl(f)["G_ema"].to(device)
print("cccccccc")
# 统计模型的参数量
total_params = sum(p.numel() for p in init_global_generator.parameters())
print(f"模型参数量总计：{total_params} 个")