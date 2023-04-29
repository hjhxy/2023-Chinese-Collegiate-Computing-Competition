import torch
from torch import nn

"""
源代码来自YOLOX自带的DWconv-深度可分离卷积
但是在这里为了与CLEM的结构设计一致，去除了激活函数和中间的归一化层
"""
class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False):
        super().__init__()
        pad         = (ksize - 1) // 2
        self.conv   = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
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
ljy的CLEM初探（结构上是一个输入输出相等的模块）

结构设计
1.感受野范围KernelSize暂定为统一尺寸
2.特征图减半的方式为削减维度，并不改变W和H的大小
3.里面的BN参数与YOlOX的源设计一致
"""
class CLEM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(CLEM, self).__init__()

        # ----主路上的设计---- #
        f_half = in_channels // 2
        self.Conv_main = BaseConv(in_channels=in_channels, out_channels=f_half, ksize=kernel_size, stride=1)
        # 主路的深度分离卷积，特征图减半（仅通过维度方向削减）
        self.bn = nn.BatchNorm2d(num_features=f_half, eps=0.001, momentum=0.03)
        self.DW_half = DWConv(in_channels=f_half, out_channels=f_half, ksize=kernel_size)

        self.BN = nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.03)
        self.RelU = nn.ReLU()

        # ----支路上的设计---- #
        # 经过计算，当扩张为 2 时，pad也为 2
        self.ConvRes_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=kernel_size, padding=2, dilation=2)
        # 同理，这里pad为 3
        self.ConvRes_2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=kernel_size, padding=3, dilation=3)
        self.BN_res = nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.03)
        self.RelU_res = nn.ReLU()

    def forward(self, input):

        # ----支路上的特征提取---- #
        res = input
        # 第一次扩张感受野
        res_1 = self.RelU_res(self.BN_res(self.ConvRes_1(res)))
        # 第二次扩张感受野，可作为输出
        res_2 = self.RelU_res(self.BN_res(self.ConvRes_2(res_1)))
        res_out = res_2

        # ----主路上的特征提取---- #
        x = input
        # 第一次卷积提取
        x_1 = self.Conv_main(x)
        x_res_1 = x_1
        # DW处理
        x_1 = self.DW_half(self.bn(x_1))
        x_out_1 = torch.cat([x_1, x_res_1], dim=1)

        # 第二次卷积提取
        x_2 = self.Conv_main(self.RelU(self.BN(x_out_1)))
        x_res_2 = x_2
        # DW处理
        x_2 = self.DW_half(self.bn(x_2))
        x_out_2 = torch.cat([x_2, x_res_2], dim=1)

        main_out = self.BN(x_out_2)

        return torch.add(res_out, main_out)



