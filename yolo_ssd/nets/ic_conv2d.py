import torch
import torch.nn as nn
import re


# 整个就大无语事件，这CVPR2021没注释就不说了，结果尼玛又是吹牛逼，json里面的参数要自己调
class ICConv2d(nn.Module):
    def __init__(self, pattern_dist, inplanes, planes, kernel_size, stride=1, groups=1, bias=False):
        super(ICConv2d, self).__init__()
        self.conv_list = nn.ModuleList()
        self.planes = planes
        for pattern in pattern_dist:
            channel = pattern_dist[pattern]
            pattern_trans = re.findall(r"\d+\.?\d*", pattern)
            pattern_trans[0] = int(pattern_trans[0]) + 1
            pattern_trans[1] = int(pattern_trans[1]) + 1
            if channel > 0:
                padding = [0, 0]
                padding[0] = (kernel_size + 2 * (pattern_trans[0] - 1)) // 2
                padding[1] = (kernel_size + 2 * (pattern_trans[1] - 1)) // 2
                self.conv_list.append(nn.Conv2d(inplanes, channel, kernel_size=kernel_size, stride=stride,
                                                padding=padding, bias=bias, groups=groups, dilation=pattern_trans))

    def forward(self, x):
        out = []
        for conv in self.conv_list:
            out.append(conv(x))

        # 这里还存在一个堆叠操作，说明ICConv本身是有对fmap进行拆解的
        out = torch.cat(out, dim=1)

        # print(out.shape[1])
        # print(self.planes)
        assert out.shape[1] == self.planes
        return out
