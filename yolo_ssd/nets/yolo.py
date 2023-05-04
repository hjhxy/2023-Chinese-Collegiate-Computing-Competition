import torch
import torch.nn as nn

from yolo_ssd.nets.backbone import Backbone, Multi_Concat_Block, Conv

from yolo_ssd.nets.biformer import BiFormerBlock
from yolo_ssd.nets.PConv import PConv

# class SPPCSPC(nn.Module):
#     # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(13, 9, 5)):
#         super(SPPCSPC, self).__init__()
#         c_ = int(2 * c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c1, c_, 1, 1)
#         self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
#         self.cv3 = Conv(4 * c_, c_, 1, 1)
#         self.cv4 = Conv(2 * c_, c2, 1, 1)
#
#     def forward(self, x):
#         x1 = self.cv1(x)
#         y1 = self.cv3(torch.cat([m(x1) for m in self.m] + [x1], 1))
#         y2 = self.cv2(x)
#         return self.cv4(torch.cat((y1, y2), dim=1))

class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(13, 9, 5)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        # self.cv1 = Conv(c1, c_, 1, 1)
        # self.cv2 = Conv(c1, c_, 1, 1)

        # 增加PConv
        self.cv1 = nn.Sequential(
            PConv(dim=c1, n_div=4, kernel_size=1),
            nn.Conv2d(c1, c_, 1, 1)
        )
        self.cv2 = nn.Sequential(
            PConv(dim=c1, n_div=4, kernel_size=1),
            nn.Conv2d(c1, c_, 1, 1)
        )
        # 分隔线

        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        # self.cv3 = Conv(4 * c_, c_, 1, 1)
        # self.cv4 = Conv(2 * c_, c2, 1, 1)

        # 增加PConv
        self.cv3 = nn.Sequential(
            PConv(dim=4 * c_, n_div=4, kernel_size=1),
            nn.Conv2d(4 * c_, c_, 1, 1)
        )
        self.cv4 = nn.Sequential(
            PConv(dim=2 * c_, n_div=4, kernel_size=1),
            nn.Conv2d(2 * c_, c2, 1, 1)
        )
        # 分隔线

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.cv3(torch.cat([m(x1) for m in self.m] + [x1], 1))
        y2 = self.cv2(x)
        return self.cv4(torch.cat((y1, y2), dim=1))

def fuse_conv_and_bn(conv, bn):
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    w_conv  = conv.weight.clone().view(conv.out_channels, -1)
    w_bn    = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    # fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape).detach())

    b_conv  = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn    = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    # fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    fusedconv.bias.copy_((torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn).detach())
    return fusedconv

#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, pretrained=False):
        super(YoloBody, self).__init__()
        #-----------------------------------------------#
        #   定义了不同yolov7-tiny的参数
        #-----------------------------------------------#
        transition_channels = 16
        block_channels      = 16
        panet_channels      = 16
        e                   = 1
        n                   = 2
        ids                 = [-1, -2, -3, -4]
        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #-----------------------------------------------#

        #---------------------------------------------------#   
        #   生成主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   80, 80, 512
        #   40, 40, 1024
        #   20, 20, 1024
        #---------------------------------------------------#
        self.backbone   = Backbone(transition_channels, block_channels, n, pretrained=pretrained)

        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")

        self.sppcspc                = SPPCSPC(transition_channels * 32, transition_channels * 16)
        # self.conv_for_P5            = Conv(transition_channels * 16, transition_channels * 8)
        # self.conv_for_feat2         = Conv(transition_channels * 16, transition_channels * 8)

        # P5：增加PConv
        self.conv_for_P5 = nn.Sequential(
            PConv(transition_channels * 16, n_div=4, kernel_size=1),
            nn.Conv2d(transition_channels * 16, transition_channels * 8, kernel_size=1)
        )
        self.conv_for_feat2 = nn.Sequential(
            PConv(transition_channels * 16, n_div=4, kernel_size=1),
            nn.Conv2d(transition_channels * 16, transition_channels * 8, kernel_size=1)
        )
        # 分隔线

        self.conv3_for_upsample1    = Multi_Concat_Block(transition_channels * 16, panet_channels * 4, transition_channels * 8, e=e, n=n, ids=ids)

        # self.conv_for_P4            = Conv(transition_channels * 8, transition_channels * 4)
        # self.conv_for_feat1         = Conv(transition_channels * 8, transition_channels * 4)

        # P4：增加PConv
        self.conv_for_P4 = nn.Sequential(
            PConv(transition_channels * 8, n_div=4, kernel_size=1),
            nn.Conv2d(transition_channels * 8, transition_channels * 4, kernel_size=1)
        )
        self.conv_for_feat1 = nn.Sequential(
            PConv(transition_channels * 8, n_div=4, kernel_size=1),
            nn.Conv2d(transition_channels * 8, transition_channels * 4, kernel_size=1)
        )
        # 分隔线

        self.conv3_for_upsample2    = Multi_Concat_Block(transition_channels * 8, panet_channels * 2, transition_channels * 4, e=e, n=n, ids=ids)

        self.down_sample1           = Conv(transition_channels * 4, transition_channels * 8, k=3, s=2)
        self.conv3_for_downsample1  = Multi_Concat_Block(transition_channels * 16, panet_channels * 4, transition_channels * 8, e=e, n=n, ids=ids)

        self.down_sample2           = Conv(transition_channels * 8, transition_channels * 16, k=3, s=2)
        self.conv3_for_downsample2  = Multi_Concat_Block(transition_channels * 32, panet_channels * 8, transition_channels * 16, e=e, n=n, ids=ids)

        # self.rep_conv_1 = Conv(transition_channels * 4, transition_channels * 8, 3, 1)
        # self.rep_conv_2 = Conv(transition_channels * 8, transition_channels * 16, 3, 1)
        # self.rep_conv_3 = Conv(transition_channels * 16, transition_channels * 32, 3, 1)

        # 这里增加PConv
        self.rep_conv_1 = nn.Sequential(
            PConv(transition_channels * 4, n_div=4),
            nn.Conv2d(transition_channels * 4, transition_channels * 8, kernel_size=1)
        )
        self.rep_conv_2 = nn.Sequential(
            PConv(transition_channels * 8, n_div=4),
            nn.Conv2d(transition_channels * 8, transition_channels * 16, kernel_size=1)
        )
        self.rep_conv_3 = nn.Sequential(
            PConv(transition_channels * 16, n_div=4),
            nn.Conv2d(transition_channels * 16, transition_channels * 32, kernel_size=1)
        )
        # 分隔线

        # 增加BiFormer内置的注意力
        # 这里head前的特征维度全部相同，所以不用额外给参数
        self.att_P3 = BiFormerBlock(dim=transition_channels * 8, outdim=transition_channels * 8, n_win=8)
        self.att_P4 = BiFormerBlock(dim=transition_channels * 16, outdim=transition_channels * 16, n_win=4)
        self.att_P5 = BiFormerBlock(dim=transition_channels * 32, outdim=transition_channels * 32, n_win=2)
        # 分隔线

        self.yolo_head_P3 = nn.Conv2d(transition_channels * 8, len(anchors_mask[2]) * (5 + num_classes), 1)
        self.yolo_head_P4 = nn.Conv2d(transition_channels * 16, len(anchors_mask[1]) * (5 + num_classes), 1)
        self.yolo_head_P5 = nn.Conv2d(transition_channels * 32, len(anchors_mask[0]) * (5 + num_classes), 1)

    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.fuseforward
        return self
    
    def forward(self, x):
        #  backbone
        feat1, feat2, feat3 = self.backbone.forward(x)
        
        P5          = self.sppcspc(feat3)
        P5_conv     = self.conv_for_P5(P5)
        P5_upsample = self.upsample(P5_conv)
        P4          = torch.cat([self.conv_for_feat2(feat2), P5_upsample], 1)
        P4          = self.conv3_for_upsample1(P4)

        P4_conv     = self.conv_for_P4(P4)
        P4_upsample = self.upsample(P4_conv)
        P3          = torch.cat([self.conv_for_feat1(feat1), P4_upsample], 1)
        P3          = self.conv3_for_upsample2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], 1)
        P4 = self.conv3_for_downsample1(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, P5], 1)
        P5 = self.conv3_for_downsample2(P5)
        
        P3 = self.rep_conv_1(P3)
        P4 = self.rep_conv_2(P4)
        P5 = self.rep_conv_3(P5)

        # 在head和feature前面增加注意力实例化
        P3 = self.att_P3(P3)
        P4 = self.att_P4(P4)
        P5 = self.att_P5(P5)
        # 分隔线

        #---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size, 75, 80, 80)
        #---------------------------------------------------#
        out2 = self.yolo_head_P3(P3)
        #---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size, 75, 40, 40)
        #---------------------------------------------------#
        out1 = self.yolo_head_P4(P4)
        #---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size, 75, 20, 20)
        #---------------------------------------------------#
        out0 = self.yolo_head_P5(P5)

        return [out0, out1, out2]
