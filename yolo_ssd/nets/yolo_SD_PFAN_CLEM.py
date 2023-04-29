#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import json

import torch
import torch.nn as nn
# from mmcv.ops import SAConv2d

from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv

# 在源SD_PFAN的基础上增加个 IC-Conv
from .darknet_CLEM import CSPDarknet_CLEM
from .ic_conv2d import ICConv2d

pattern_path = 'E:/aproject/yolo_ssd/nets/ic_resnet50_k9.json'
pattern = None
# 这里选择 15 是因为该代码并不具备NAS自动搜索功能，但从 json 文件中提供了通道字典，需要手动操作一下，选择相应层级
pattern_index = 15


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width=1.0, in_channels=[256, 512, 1024], act="silu", depthwise=False, ):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(in_channels=int(in_channels[i] * width), out_channels=int(256 * width), ksize=1, stride=1,
                         act=act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=num_classes, kernel_size=1, stride=1, padding=0)
            )

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act),
                Conv(in_channels=int(256 * width), out_channels=int(256 * width), ksize=3, stride=1, act=act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=4, kernel_size=1, stride=1, padding=0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels=int(256 * width), out_channels=1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, inputs):
        # ---------------------------------------------------#
        #   inputs输入
        #   P3_out  80, 80, 256
        #   P4_out  40, 40, 512
        #   P5_out  20, 20, 1024
        # ---------------------------------------------------#
        outputs = []
        for k, x in enumerate(inputs):
            # ---------------------------------------------------#
            #   利用1x1卷积进行通道整合
            # ---------------------------------------------------#
            x = self.stems[k](x)
            # ---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            # ---------------------------------------------------#
            cls_feat = self.cls_convs[k](x)
            # ---------------------------------------------------#
            #   判断特征点所属的种类
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            # ---------------------------------------------------#
            cls_output = self.cls_preds[k](cls_feat)

            # ---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            # ---------------------------------------------------#
            reg_feat = self.reg_convs[k](x)
            # ---------------------------------------------------#
            #   特征点的回归系数
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            # ---------------------------------------------------#
            reg_output = self.reg_preds[k](reg_feat)
            # ---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            # ---------------------------------------------------#
            obj_output = self.obj_preds[k](reg_feat)

            output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs


"""
（1）算法细节：
    1.相比 PANet ，这个结构卷积模块并没有那么多
    2.在ljy的代码中，最后一层 head 前面的特征提取并没有去除，实验如果效果不好，可以保留最后一层的CSPLayer
    3.有个用于YOLOX存在一个bug，输出通道并不能与前面对齐。
      所以这里要用一个模块重新resize一下，考虑用1x1卷积先做一个尝试，这里用YOLOX中的CBS块
（2）代码细节：
    1.为了命名方便，先按照结构图从 forward 写起，然后再补 init，在forward的构建中，先从高层开始构建，但是代码是相反排布的
    2.按通道维度堆叠，网络的维度格式为，(batchsize, channels, w, h) ，所以维度设置为 1
    3.参数量是按 YOLOX-s 尺寸写的，但是设置里面要留 多尺度模型的接口。
    4.卷积模块的参数均为 ksize=3, stride=2，但在使用过程中注意pad
    5.mmdetection中有些包不一定能够找到，只是存在于内置文件中
"""

"""
现将 SAC 换成 ICConv 进行初步尝试
"""


class YOLO_SD_PFAN(nn.Module):
    def __init__(self, depth=1.0, width=1.0, in_features=("dark3", "dark4", "dark5"), in_channels=[256, 512, 1024],
                 depthwise=False, act="silu"):
        super().__init__()

        # 这里定义 ICConv 的全局模式字
        global pattern, pattern_index, pattern_path
        # 加载 模式关键字
        with open(pattern_path, 'r') as fin:
            pattern = json.load(fin)

        Conv = DWConv if depthwise else BaseConv
        self.backbone = CSPDarknet_CLEM(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features

        # 开始处理特征提取过程

        # 底层到中层的 上采样
        # 512, 20, 20 -> 256, 40, 40
        self.upsample_L_to_M = nn.ConvTranspose2d(
            in_channels=int(in_channels[2] * width),
            out_channels=int(in_channels[1] * width),
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )

        # 中层到高层的 上采样
        # 512, 40, 40 -> 128, 80, 80
        self.upsample_M_to_H = nn.ConvTranspose2d(
            in_channels=int(2 * in_channels[1] * width),
            out_channels=int(in_channels[1] // 2 * width),
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )

        # 中层的过程特征 下采样
        # 512, 40, 40 -> 512, 20, 20
        # self.downsample_M_to_L = SAConv2d(
        #     in_channels=int(2 * in_channels[1] * width),
        #     out_channels=int(in_channels[2] * width),
        #     kernel_size=3,
        #     stride=2,
        #     padding=1
        # )

        # 引入ICConv作为下采样
        # pattern_index = pattern_index + 1
        self.downsample_M_to_L = ICConv2d(
            pattern[pattern_index],
            inplanes=int(2 * in_channels[1] * width),
            planes=int(in_channels[2] * width),
            kernel_size=3,
            stride=2
        )

        # 高层输出特征层下采样
        # 256, 80, 80 -> 512, 40, 40
        # self.downsample_H_to_M = SAConv2d(
        #     in_channels=int(2 * in_channels[0] * width),
        #     out_channels=int(2 * in_channels[1] * width),
        #     kernel_size=3,
        #     stride=2,
        #     padding=1
        # )

        # 引入ICConv作为下采样
        # pattern_index = pattern_index + 1
        self.downsample_H_to_M = ICConv2d(
            pattern[pattern_index],
            inplanes=int(2 * in_channels[0] * width),
            planes=int(2 * in_channels[1] * width),
            kernel_size=3,
            stride=2
        )

        # 通道数 resize
        # 高层；256, 80, 80 -> 128, 80, 80
        self.resize_H = BaseConv(
            in_channels=int(2 * in_channels[0] * width),
            out_channels=int(in_channels[0] * width),
            ksize=1,
            stride=1,
            act=act
        )

        # 中层：1024, 40, 40 -> 256, 40, 40
        self.resize_M = BaseConv(
            in_channels=int(4 * in_channels[1] * width),
            out_channels=int(in_channels[1] * width),
            ksize=1,
            stride=1,
            act=act
        )

        # 底层：1024, 20, 20 -> 512, 20, 20
        self.resize_L = BaseConv(
            in_channels=int(2 * in_channels[2] * width),
            out_channels=int(in_channels[2] * width),
            ksize=1,
            stride=1,
            act=act
        )

    def forward(self, input):
        out_features = self.backbone.forward(input)
        [feat1, feat2, feat3] = [out_features[f] for f in self.in_features]

        # 开始构建不同特征层，代码写法需要倒过来，但是可以从高层开始写
        # 其中，feat1~3 分别为 dark3~5，也即高层到底层，通道数分别为 128, 256, 512

        # 底层骨干特征上采样
        # 512, 20, 20 -> 256, 40, 40
        feat3_upsample = self.upsample_L_to_M(feat3)

        # 中层的过程特征
        # 256, 40, 40 + 256, 40, 40 -> 512, 40, 40,
        Fmap_M_prs = torch.cat([feat2, feat3_upsample], 1)

        # 中层的过程特征 上采样
        # 512, 40, 40 -> 128, 80, 80
        Fmap_M_prs_upsample = self.upsample_M_to_H(Fmap_M_prs)

        # 高层输出
        # 128, 80, 80 + 128, 80, 80 -> 256, 80, 80
        Fmap_H_out = torch.cat([feat1, Fmap_M_prs_upsample], 1)

        # 中层的过程特征 下采样
        # 512, 40, 40 -> 512, 20, 20
        Fmap_M_prs_downsample = self.downsample_M_to_L(Fmap_M_prs)

        # 高层输出特征层下采样
        # 256, 80, 80 -> 512, 40, 40
        Fmap_H_out_downsample = self.downsample_H_to_M(Fmap_H_out)

        # 中层输出
        # 512, 40, 40 + 512, 40, 40 -> 1024, 40, 40
        Fmap_M_out = torch.cat([Fmap_H_out_downsample, Fmap_M_prs], 1)

        # 底层输出
        # 512, 20, 20 + 512, 20, 20 -> 1024, 20, 20
        Fmap_L_out = torch.cat([feat3, Fmap_M_prs_downsample], 1)

        # 输出降维通道数
        # 高层；256, 80, 80 -> 128, 80, 80
        Fmap_H_out = self.resize_H(Fmap_H_out)

        # 中层：1024, 40, 40 -> 256, 40, 40
        Fmap_M_out = self.resize_M(Fmap_M_out)

        # 底层：1024, 20, 20 -> 512, 20, 20
        Fmap_L_out = self.resize_L(Fmap_L_out)

        # 输出返回参数从 高层向底层
        return (Fmap_H_out, Fmap_M_out, Fmap_L_out)


class YoloBody(nn.Module):
    def __init__(self, num_classes, phi):
        super().__init__()
        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33, }
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }
        depth, width = depth_dict[phi], width_dict[phi]
        depthwise = True if phi == 'nano' else False

        self.backbone = YOLO_SD_PFAN(depth, width, depthwise=depthwise)
        self.head = YOLOXHead(num_classes, width, depthwise=depthwise)

    def forward(self, x):
        fpn_outs = self.backbone.forward(x)
        outputs = self.head.forward(fpn_outs)
        return outputs
