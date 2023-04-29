import torch
from torch import nn
import torch.nn.functional as F

from yolo_ssd.nets.CSC_Layer import DictConv2d
from yolo_ssd.nets.SoftPool import SoftPooling2D

"""
源代码来自YOLOX自带的DWconv-深度可分离卷积
但是在这里为了与CLEM的结构设计一致，去除了激活函数和中间的归一化层
"""


# class h_sigmoid(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_sigmoid, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace)
#
#     def forward(self, x):
#         return self.relu(x + 3) / 6


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups,
                              bias=bias)
        # self.bn     = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        # self.act    = get_activation(act, inplace=True)

    def forward(self, x):
        return self.conv(x)


class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels)
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


"""
A5版本的CLEM
1.主路上换成DW卷积
2.旁路增加Dropout
3.主路增加DFC，不保留Res减去梯度爆炸
4.DFC中利用SoftPool
"""

class CLEM_A5(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(CLEM_A5, self).__init__()

        # ----主路上的设计---- #
        self.out_channerls = out_channels
        f_half = in_channels // 2
        self.DW_main = DWConv(in_channels=in_channels, out_channels=f_half, ksize=kernel_size, stride=1)
        # 主路的深度分离卷积，特征图减半（仅通过维度方向削减）
        self.bn = nn.BatchNorm2d(num_features=f_half, eps=0.001, momentum=0.03)
        self.DW_half = DWConv(in_channels=f_half, out_channels=f_half, ksize=kernel_size)

        self.BN = nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.03)
        self.RelU = nn.ReLU()

        # 主路上增加DFC注意力机制-来自GhostNetv2
        self.DFC_attention = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.gate_fn = nn.Sigmoid()
        self.down_sample = SoftPooling2D(kernel_size=2, strides=2)

        # ----支路上的设计---- #
        # 旁路经过一个CSC-Layer，卷积核为5x5，这里将pad=1，来平衡尺寸
        self.ConvRes_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=kernel_size, padding=2, dilation=2)
        # 同理，这里pad为 3
        self.ConvRes_2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=kernel_size, padding=3, dilation=3)
        self.BN_res = nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.03)
        self.RelU_res = nn.ReLU()
        self.Dropout = nn.Dropout(p=0.3)

    def forward(self, input):
        # ----支路上的特征提取---- #
        res = input
        # 第一次扩张感受野
        res_1 = self.RelU_res(self.BN_res(self.ConvRes_1(res)))
        # 第二次扩张感受野，可作为输出
        res_2 = self.RelU_res(self.BN_res(self.ConvRes_2(res_1)))
        res_out = self.Dropout(res_2)

        # ----主路上的特征提取---- #
        x = input
        # 第一次卷积提取
        x_1 = self.DW_main(x)
        x_res_1 = x_1
        # DW处理
        x_1 = self.DW_half(self.bn(x_1))
        x_out_1 = torch.cat([x_1, x_res_1], dim=1)

        # 第二次卷积提取
        x_2 = self.DW_main(self.RelU(self.BN(x_out_1)))
        x_res_2 = x_2
        # DW处理
        x_2 = self.DW_half(self.bn(x_2))
        x_out_2 = torch.cat([x_2, x_res_2], dim=1)

        # 在这里增加对中间输出的DFC注意力
        DFC_out = self.DFC_attention(self.down_sample(x))
        x_out_2 = x_out_2[:, :self.out_channerls, :, :] * F.interpolate(self.gate_fn(DFC_out), size=x_out_2.shape[-1],
                                                                        mode='nearest')
        main_out = self.RelU(self.BN(x_out_2))

        return torch.add(res_out, main_out)
