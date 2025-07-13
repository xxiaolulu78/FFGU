# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import math


# https://github.com/jackfrued/Python-1/blob/master/analysis/compression_analysis/psnr.py
# 函数psnr来计算两个图像（原始图像和对比图像）之间的峰值信噪比（PSNR）。
# PSNR是衡量图像质量的指标，数值越高表示图像质量越好。
def psnr(original, contrast):
    mse = np.mean((original - contrast) ** 2) / 3
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    PSNR = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return PSNR


def DLG(net, origin_grad, target_inputs):
    criterion = torch.nn.MSELoss()
    cnt = 0
    psnr_val = 0
    for idx, (gt_data, gt_out) in enumerate(target_inputs):
        # generate dummy data and label
        dummy_data = torch.randn_like(gt_data, requires_grad=True)
        dummy_out = torch.randn_like(gt_out, requires_grad=True)

        optimizer = torch.optim.LBFGS([dummy_data, dummy_out])

        history = [gt_data.data.cpu().numpy(), F.sigmoid(dummy_data).data.cpu().numpy()]
        for iters in range(100):
            def closure():

                optimizer.zero_grad()  # 清除当前的梯度。

                dummy_pred = net(F.sigmoid(dummy_data))  # 计算模型的预测输出和损失。
                dummy_loss = criterion(dummy_pred, dummy_out)  # 计算损失相对于模型参数的梯度。
                dummy_grad = torch.autograd.grad(dummy_loss, net.parameters(),
                                                 create_graph=True)  # 计算原始梯度和当前梯度之间的差异，并将这个差异反向传播。

                grad_diff = 0  # 初始化梯度差异的累积变量grad_diff为0
                for gx, gy in zip(dummy_grad, origin_grad):  # 遍历原始梯度和当前梯度
                    if gx is not None and gy is not None:  # 检查当前梯度是否为
                        grad_diff += ((gx - gy) ** 2).sum()  # 对于每一对gx和gy：计算两个梯度之间的差值gx - gy。
                        # 对差值进行平方(gx - gy) ** 2，得到差值的平方。
                        # 通过.sum()函数计算所有元素的平方和，得到该对梯度差异的总和。 将这个总和加到grad_diff上，进行累积。
                grad_diff.backward()  # 将累积的梯度差异反向传播回模型

                return grad_diff

            optimizer.step(closure)

        # plt.figure(figsize=(3*len(history), 4))
        # for i in range(len(history)):
        #     plt.subplot(1, len(history), i + 1)
        #     plt.imshow(history[i])
        #     plt.title("iter=%d" % (i * 10))
        #     plt.axis('off')

        # plt.savefig(f'dlg_{algo}_{cid}_{idx}' + '.pdf', bbox_inches="tight")

        history.append(F.sigmoid(dummy_data).data.cpu().numpy())  # 将历史数据（即之前计算得到的PSNR值）添加到history列表中

        p = psnr(history[0], history[2])  # 计算原始图像和压缩图像之间的PSNR值
        if not math.isnan(p):
            psnr_val += p  # 将PSNR值累加到psnr_val变量中
            cnt += 1  # 将计数器cnt加1

    if cnt > 0:
        return psnr_val / cnt
    else:
        return None
