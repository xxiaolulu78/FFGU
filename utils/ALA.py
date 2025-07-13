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

import numpy as np
import torch
import torch.nn as nn
import copy
import random
from torch.utils.data import DataLoader
from typing import List, Tuple


class ALA:
    def __init__(self,
                 cid: int,
                 loss: nn.Module,
                 train_data: List[Tuple],
                 batch_size: int,
                 rand_percent: int,
                 layer_idx: int = 0,
                 eta: float = 1.0,
                 device: str = 'cpu',
                 threshold: float = 0.1,
                 num_pre_loss: int = 10) -> None:
        """
        Initialize ALA module

        Args:
            cid: Client ID. 
            loss: The loss function. 
            train_data: The reference of the local training data.
            batch_size: Weight learning batch size.
            rand_percent: The percent of the local training data to sample.
            layer_idx: Control the weight range. By default, all the layers are selected. Default: 0
            eta: Weight learning rate. Default: 1.0
            device: Using cuda or cpu. Default: 'cpu'
            threshold: Train the weight until the standard deviation of the recorded losses is less than a given threshold. Default: 0.1
            num_pre_loss: The number of the recorded losses to be considered to calculate the standard deviation. Default: 10

        Returns:
            None.
        """

        self.cid = cid
        self.loss = loss
        self.train_data = train_data
        self.batch_size = batch_size
        self.rand_percent = rand_percent
        self.layer_idx = layer_idx
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.device = device

        self.weights = None  # Learnable local aggregation weights.
        self.start_phase = True

    def adaptive_local_aggregation(self,
                                   global_model: nn.Module,
                                   local_model: nn.Module) -> None:  # 接收全局模型和本地模型作为参数
        """
        Generates the Dataloader for the randomly sampled local training data and 
        preserves the lower layers of the update. 

        Args:
            global_model: The received global/aggregated model. 
            local_model: The trained local model. 

        Returns:
            None.
        """

        # 随机采样部分本地训练数据。
        rand_ratio = self.rand_percent / 100
        rand_num = int(rand_ratio * len(self.train_data))
        rand_idx = random.randint(0, len(self.train_data) - rand_num)
        rand_loader = DataLoader(self.train_data[rand_idx:rand_idx + rand_num], self.batch_size, drop_last=False)

        # 获取模型的参数引用
        params_g = list(global_model.parameters())
        params = list(local_model.parameters())

        # 在第一次通信迭代时将ALA（激活层）关闭
        if torch.sum(params_g[0] - params[0]) == 0:
            return

        # 将全局模型的低层参数复制到本地模型，以保留这些层的更新
        for param, param_g in zip(params[:-self.layer_idx], params_g[:-self.layer_idx]):
            param.data = param_g.data.clone()

        # 创建一个临时模型用于权重学习，冻结低层的权重。
        model_t = copy.deepcopy(local_model)  # 复制本地模型local_model到model_t，以便在权重学习过程中使用，而不改变原始本地模型。
        params_t = list(model_t.parameters())

        # only consider higher layers
        params_p = params[-self.layer_idx:]  # 本地模型higher layers
        params_gp = params_g[-self.layer_idx:]  # 全局模型higher layers
        params_tp = params_t[-self.layer_idx:]  # 临时模型higher layers

        # 将临时模型中低层的requires_grad属性设置为False，这样在反向传播时不会计算这些层的梯度。
        for param in params_t[:-self.layer_idx]:
            param.requires_grad = False

        # 创建一个SGD优化器，用于计算高层层间的梯度。
        # 学习率设置为0，因为我们只关心梯度的方向，而不关心步长。
        optimizer = torch.optim.SGD(params_tp, lr=0)

        #将其初始化为与params_p相同形状的全1张量
        if self.weights == None:
            self.weights = [torch.ones_like(param.data).to(self.device) for param in params_p]

        # initialize the higher layers in the temp local model
        for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp,
                                                   self.weights):
            param_t.data = param + (param_g - param) * weight

        # weight learning
        losses = []
        cnt = 0  # weight training iteration counter
        while True:
            for x, y in rand_loader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                output = model_t(x)
                loss_value = self.loss(output, y)  # 执行前向传播，计算损失值loss_value
                loss_value.backward()  #执行反向传播，计算梯度

                # 更新权重：根据高层层间的梯度来调整权重，使用torch.clamp确保权重在0和1之间
                for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                           params_gp, self.weights):
                    weight.data = torch.clamp(
                        weight - self.eta * (param_t.grad * (param_g - param)), 0, 1)

                # 更新临时模型的参数：使用更新后的权重和全局模型的参数差异来调整临时模型的参数
                for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                           params_gp, self.weights):
                    param_t.data = param + (param_g - param) * weight

            losses.append(loss_value.item())  #记录每次迭代的损失值，并在满足条件时停止训练
            cnt += 1

            # only train one epoch in the subsequent iterations
            if not self.start_phase:
                break

            # train the weight until convergence
            if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold:
                print('Client:', self.cid, '\tStd:', np.std(losses[-self.num_pre_loss:]),
                      '\tALA epochs:', cnt)
                break

        self.start_phase = False  #结束初始权重学习阶段

        # obtain initialized local model
        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone()
