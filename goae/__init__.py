import torch


import torch.nn.functional as F

from torch import nn, Tensor
from enum import Enum
from .swin_transformer import build_model
from typing import Optional

class ProgressiveStage(Enum):
    WTraining = 0
    Delta1Training = 1
    Delta2Training = 2
    Delta3Training = 3
    Delta4Training = 4
    Delta5Training = 5
    Delta6Training = 6
    Delta7Training = 7
    Delta8Training = 8
    Delta9Training = 9
    Delta10Training = 10
    Delta11Training = 11
    Delta12Training = 12
    Delta13Training = 13
    Inference = 14

# 构建一个由多个线性层和LeakyReLU激活函数组成的序列模型
def get_mlp_layer(in_dim, out_dim, mlp_layer=2):
    module_list = nn.ModuleList()  # 创建一个模块列表 存储线性层和激活层
    for j in range(mlp_layer - 1):
        module_list.append(nn.Linear(in_dim, in_dim))
        module_list.append(nn.LeakyReLU())
    module_list.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*module_list)  # 使用nn.Sequential将所有层打包成一个顺序模型，并返回这个模型。

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

#交叉注意力机制
class CrossAttention(nn.Module):
    def __init__(
        self,
        d_model,  # 输入维度 输出维度
        nhead,  # 多头注意力机制的头数
        dim_feedforward=2048, # 前馈神经网络隐藏层维度
        dropout=0.1,
        activation="relu",
        normalize_before=False, # 是否在每一层之前进行归一化
        batch_first=True, # 是否将batch维度放在第一个位置
    ):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # 初始化两个层归一化层 norm1 和 norm2
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)

        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        # 根据 activation 参数获取相应的激活函数
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(
        self,
        tgt,  #查询张量
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ):
        tgt2 = self.multihead_attn(
            query=tgt,
            key=memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class GOAEncoder(nn.Module):
    def __init__(
        self, swin_config, mlp_layer=2, ws_dim=14, stage_list=[20000, 40000, 60000]
    ):
        super(GOAEncoder, self).__init__()
        self.style_count = ws_dim  # 风格向量的维度，默认为 14
        self.stage_list = stage_list  # 一个整数列表，表示不同训练阶段迭代次数的阈值。
        self.stage_dict = {"base": 0, "coarse": 1, "mid": 2, "fine": 3}
        self.stage = 3

        ## -------------------------------------------------- base w0 swin transformer -------------------------------------------
        self.swin_model = build_model(swin_config)  #这是一个基于 Swin Transformer 的模型，用于提取特征。
        # 定义了多个 MLP 层，用于空间和通道特征的映射
        self.mapper_base_spatial = get_mlp_layer(64, 1, mlp_layer)
        self.mapper_base_channel = get_mlp_layer(1024, 512, mlp_layer)
        #使用 nn.AdaptiveMaxPool1d 创建最大池化层，降低特征维度
        self.maxpool_base = nn.AdaptiveMaxPool1d(1)

        ## -------------------------------------------------- w Query mapper coarse mid fine  1024*64 -> (4-1)*512 3*512 7*512 -------------------------------------------
        self.maxpool_query = nn.AdaptiveMaxPool1d(1)
        #定义了多个线性层和 LeakyReLU 激活函数的序列，用于进一步的特征转换
        self.mapper_query_spatial_coarse = get_mlp_layer(64, 3, mlp_layer)
        self.mapper_query_channel_coarse = get_mlp_layer(1024, 512, mlp_layer)
        self.mapper_query_spatial_mid = get_mlp_layer(64, 3, mlp_layer)
        self.mapper_query_channel_mid = get_mlp_layer(1024, 512, mlp_layer)
        self.mapper_query_spatial_fine = get_mlp_layer(64, 7, mlp_layer)
        self.mapper_query_channel_fine = get_mlp_layer(1024, 512, mlp_layer)

        ## -------------------------------------------------- w KQ coarse mid fine mapper to 512 -------------------------
        self.mapper_coarse_channel = nn.Sequential(nn.Linear(512, 512), nn.LeakyReLU())
        self.mapper_mid_channel = nn.Sequential(nn.Linear(256, 512), nn.LeakyReLU())
        self.mapper_fine_channel = nn.Sequential(
            nn.Linear(128, 256), nn.LeakyReLU(), nn.Linear(256, 512), nn.LeakyReLU()
        )

        self.mapper_coarse_to_mid_spatial = nn.Sequential(
            nn.Linear(256, 512), nn.LeakyReLU(), nn.Linear(512, 1024), nn.LeakyReLU()
        )
        self.mapper_mid_to_fine_spatial = nn.Sequential(
            nn.Linear(1024, 2048), nn.LeakyReLU(), nn.Linear(2048, 4096), nn.LeakyReLU()
        )
        # 初始化了三个 CrossAttention 模块，分别对应粗（coarse）、中（mid）、细（fine）三个不同粒度的注意力机制
        ## -------------------------------------------------- w KQ coarse mid fine Cross Attention -------------------------
        self.cross_att_coarse = CrossAttention(512, 4, 1024, batch_first=True)
        self.cross_att_mid = CrossAttention(512, 4, 1024, batch_first=True)
        self.cross_att_fine = CrossAttention(512, 4, 1024, batch_first=True)
        self.progressive_stage = ProgressiveStage.Inference
    # 根据当前迭代次数 iter 设置模型的训练阶段
    def set_stage(self, iter):
        if iter > self.stage_list[-1]:
            self.stage = 3
        else:
            for i, stage_iter in enumerate(self.stage_list):
                if iter < stage_iter:
                    break
            self.stage = i

        print(f"change training stage to {self.stage}")

    def forward(self, x):
        B = x.shape[0]
        # 使用swin_model提取不同粒度的特征：基础（base）、查询（query）、粗（coarse）、中（mid）、细（fine）
        x_base, x_query, x_coarse, x_mid, x_fine = self.swin_model(x)

        ## ----------------------  base 对基础特征使用最大池化和 MLP 层来生成初始的风格编码 ws_base
        ws_base_max = self.maxpool_base(x_base).transpose(1, 2)
        ws_base_linear = self.mapper_base_spatial(x_base)
        ws_base = self.mapper_base_channel(ws_base_linear.transpose(1, 2) + ws_base_max)

        ws_base = ws_base.repeat(1, 14, 1)
        # 如果当前阶段是基础阶段（"base"），则直接将 ws_base 作为最终输出
        if self.stage == self.stage_dict["base"]:
            ws = ws_base
            return ws, ws_base

        ## ------------------------ coarse mid fine ---  query

        ws_query_max = self.maxpool_query(x_query).transpose(1, 2)

        if self.stage >= self.stage_dict["coarse"]:
            ws_query_linear_coarse = self.mapper_query_spatial_coarse(x_query)  #首先使用空间映射器处理查询特征
            ws_query_coarse = self.mapper_query_channel_coarse(
                ws_query_linear_coarse.transpose(1, 2) + ws_query_max
            )  #使用通道映射器生成该阶段的风格编码

            if self.stage >= self.stage_dict["mid"]:
                ws_query_linear_mid = self.mapper_query_spatial_mid(x_query)
                ws_query_mid = self.mapper_query_channel_mid(
                    ws_query_linear_mid.transpose(1, 2) + ws_query_max
                )

                if self.stage >= self.stage_dict["fine"]:
                    ws_query_linear_fine = self.mapper_query_spatial_fine(x_query)
                    ws_query_fine = self.mapper_query_channel_fine(
                        ws_query_linear_fine.transpose(1, 2) + ws_query_max
                    )

        ## -------------------------  carse, mid, fine -----  key-value
        if self.stage >= self.stage_dict["coarse"]:
            kv_coarse = self.mapper_coarse_channel(x_coarse)

            if self.stage >= self.stage_dict["mid"]:
                kv_mid = self.mapper_mid_channel(
                    x_mid
                ) + self.mapper_coarse_to_mid_spatial(
                    kv_coarse.transpose(1, 2)
                ).transpose(
                    1, 2
                )

                if self.stage >= self.stage_dict["fine"]:
                    kv_fine = self.mapper_fine_channel(
                        x_fine
                    ) + self.mapper_mid_to_fine_spatial(
                        kv_mid.transpose(1, 2)
                    ).transpose(
                        1, 2
                    )

        ## ------------------------- carse, mid, fine -----  Cross attention
        #对于每个阶段，将注意力输出与其他零张量拼接，构建增量风格编码 ws_delta。零张量的维度根据阶段不同而变化
        if self.stage >= self.stage_dict["coarse"]:
            ws_coarse = self.cross_att_coarse(ws_query_coarse, kv_coarse)
            zero_1 = torch.zeros(B, 1, 512).to(ws_base.device)
            zero_2 = torch.zeros(B, 10, 512).to(ws_base.device)
            ws_delta = torch.cat([zero_1, ws_coarse, zero_2], dim=1)

            if self.stage >= self.stage_dict["mid"]:
                ws_mid = self.cross_att_mid(ws_query_mid, kv_mid)
                zero_1 = torch.zeros(B, 1, 512).to(ws_base.device)
                zero_2 = torch.zeros(B, 7, 512).to(ws_base.device)
                ws_delta = torch.cat([zero_1, ws_coarse, ws_mid, zero_2], dim=1)

                if self.stage >= self.stage_dict["fine"]:
                    ws_fine = self.cross_att_fine(ws_query_fine, kv_fine)

                    zero = torch.zeros(B, 1, 512).to(ws_base.device)

                    ws_delta = torch.cat([zero, ws_coarse, ws_mid, ws_fine], dim=1)

        ws = ws_base + ws_delta  #基础风格编码 ws_base 和增量编码 ws_delta 相加，得到最终的风格编码 ws
        return ws, ws_base  #返回风格编码和基础风格编码
    
if __name__ == "__main__":
    # from .training.triplane import TriplaneGenerator
    import os
    import sys
    sys.path.append(os.path.abspath("."))
    import legacy
    import dnnlib
    import torch.nn.functional as F
    import numpy as np
    from argparse import ArgumentParser
    from PIL import Image
    from torchvision import transforms
    from swin_config import get_config
    from camera_utils import LookAtPoseSampler, FOV_to_intrinsics

    def tensor_to_image(t):  #将PyTorch张量转换为PIL图像
        t = (t.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return Image.fromarray(t[0].cpu().numpy(), "RGB")

    def image_to_tensor(i, size=256):  #PIL图像转换为PyTorch张量，用于神经网络处理。
        i = i.resize((size, size))
        i = np.array(i)
        i = i.transpose(2, 0, 1)
        i = torch.from_numpy(i).to(torch.float32).to("cuda") / 127.5 - 1
        return i
    
    parser = ArgumentParser()
    parser.add_argument("--encoder_ckpt", type=str, default="./files/encoder_FFHQ.pt")
    parser.add_argument("--img", type=str, default="fake_images/adj_4_std_0.5_seed_0_lr_2e-4/pretrained/1721.png")
    args = parser.parse_args()

    goae = GOAEncoder(swin_config=get_config(), stage_list=[10000, 20000, 30000])  #创建GOAEncoder实例
    goae.load_state_dict(torch.load(args.encoder_ckpt, map_location="cuda"))  #加载预训练权重到编码器模型
    goae = goae.to("cuda")
    # goae.eval()
    img = Image.open(args.img).convert("RGB")
    img = image_to_tensor(img).unsqueeze(0)
    w, _ = goae(img)  #将处理后的张量img传递给goae模型 该模型计算图像的潜在空间表示
    print(w.shape)

    w = w + torch.load("./files/w_avg_ffhqrebalanced512-128.pt").unsqueeze(0).to("cuda")  #使用torch.load加载预训练的平均潜在向量 使用unsqueeze(0)添加批次维度，使其可以与编码器输出的潜在向量w结合

    with dnnlib.util.open_url("./files/ffhqrebalanced512-128.pkl") as f:
        g_source = legacy.load_network_pkl(f)["G_ema"].to("cuda")  #加载预训练的生成器模型并获取其中的G_ema（指数移动平均生成器）
    # 定义相机的内参（intrinsics），这通常包括焦距、主点坐标等
    # intrinsics = FOV_to_intrinsics(18.837, device="cuda") # from eg3d
    intrinsics = torch.tensor([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1], device="cuda") # from goae


    # cam_pivot = torch.tensor(g_source.rendering_kwargs.get("avg_cam_pivot", [0, 0, 0]), device="cuda") # from eg3d
    cam_pivot = torch.tensor([0, 0, 0.2], device="cuda") # from goae
#使用LookAtPoseSampler和cam_pivot（相机中心点）设置相机的姿态，这决定了相机观察3D场景的方向和位置
    cam_radius = g_source.rendering_kwargs.get("avg_cam_radius", 2.7)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi / 2, np.pi / 2, cam_pivot, radius=cam_radius, device="cuda")
    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
    # 使用生成的潜在向量w和相机的内参，通过生成器模型g_source生成图像
    # 定义angle_p（俯仰角）和angle_y（偏航角），这些角度决定了相机相对于目标对象的观察方向
    angle_p = -0.2
    # angle_y = np.random.uniform(-np.pi / 4, np.pi / 4) 
    angle_y = 0
    cam2world_pose = LookAtPoseSampler.sample(np.pi / 2 + angle_y, np.pi / 2 + angle_p, cam_pivot, radius=cam_radius, device="cuda")
    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], dim=1)
    # 使用生成器模型的synthesis方法和潜在向量w以及相机参数camera_params来生成图像
    img = g_source.synthesis(w, camera_params)["image"]
    img = tensor_to_image(img)
    img.save("recon.png")
