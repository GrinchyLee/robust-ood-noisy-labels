# -*- coding: UTF-8 -*-
'''
@Project : ProPos 
@File    : __init__.py
@Author  : Zhizhong Huang from Fudan University
@Homepage: https://hzzone.github.io/
@Email   : zzhuang19@fudan.edu.cn
@Date    : 2022/10/19 9:25 PM 
'''

import sys
import os
import torch.nn as nn
import torch.nn.functional as F

from . import resnet, preact_resnet

_openood_path = os.path.join(os.path.dirname(__file__), '..', '..', 'openood')
_openood_path = os.path.abspath(_openood_path)
if _openood_path not in sys.path:
    sys.path.insert(0, _openood_path)

from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.networks.resnet18_64x64 import ResNet18_64x64


class _ResNet18_32x32_Encoder(ResNet18_32x32):
    """ResNet18_32x32의 backbone만 사용해 512d feature를 반환.
    layer 이름이 ResNet18_32x32와 동일해 OOD 평가 시 바로 로드 가능.
    """
    def __init__(self, num_classes=10):
        super().__init__(num_classes=num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        return out.view(out.size(0), -1)


class _ResNet18_64x64_Encoder(ResNet18_64x64):
    """ResNet18_64x64의 backbone만 사용해 512d feature를 반환.
    layer 이름이 ResNet18_64x64와 동일해 OOD 평가 시 바로 로드 가능.
    """
    def __init__(self, num_classes=10):
        super().__init__(num_classes=num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        return out.view(out.size(0), -1)


backbone_dict = {
    'bigresnet18': [resnet.ResNet('resnet18', cifar=True), 512],
    'bigresnet34': [resnet.ResNet('resnet18', cifar=True), 512],
    'bigresnet50': [resnet.ResNet('resnet18', cifar=True), 2048],
    'bigresnet18_preact': [preact_resnet.ResNet18, 512],
    'resnet18': [_ResNet18_32x32_Encoder, 512],
    'resnet18_64x64': [_ResNet18_64x64_Encoder, 512],
    'resnet34': [resnet.ResNet('resnet34'), 512],
    'resnet50': [resnet.ResNet('resnet50'), 2048],
}
