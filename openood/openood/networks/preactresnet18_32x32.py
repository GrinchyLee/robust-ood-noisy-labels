import torch.nn as nn
import torch.nn.functional as F


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if len(self.shortcut) != 0 else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActResNet18_32x32(nn.Module):
    def __init__(self, block=PreActBasicBlock, num_blocks=None, num_classes=10):
        super(PreActResNet18_32x32, self).__init__()
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.feature_size = 512 * block.expansion

        feat_dim = 128
        dim_in = self.feature_size
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim)
        )
        self.pseudo_linear = nn.Linear(dim_in, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_feature=False, return_feature_list=False, train=False, use_ph=False):
        feature1 = F.relu(self.bn1(self.conv1(x)))
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)
        feature5 = self.avgpool(feature5)
        feature = feature5.view(feature5.size(0), -1)
        logits_cls = self.fc(feature)
        feature_list = [feature1, feature2, feature3, feature4, feature5]

        if train:
            feat_c = self.head(feature)
            if use_ph:
                out_linear_debias = self.pseudo_linear(feature)
                if return_feature:
                    return logits_cls, out_linear_debias, F.normalize(feat_c, dim=1), feature
                elif return_feature_list:
                    return logits_cls, out_linear_debias, F.normalize(feat_c, dim=1), feature_list
                else:
                    return logits_cls, out_linear_debias, F.normalize(feat_c, dim=1)
            else:
                if return_feature:
                    return logits_cls, F.normalize(feat_c, dim=1), feature
                elif return_feature_list:
                    return logits_cls, F.normalize(feat_c, dim=1), feature_list
                else:
                    return logits_cls, F.normalize(feat_c, dim=1)
        else:
            if use_ph:
                out_linear_debias = self.pseudo_linear(feature)
                if return_feature:
                    return logits_cls, out_linear_debias, feature
                elif return_feature_list:
                    return logits_cls, out_linear_debias, feature_list
                else:
                    return logits_cls, out_linear_debias
            else:
                if return_feature:
                    return logits_cls, feature
                elif return_feature_list:
                    return logits_cls, feature_list
                else:
                    return logits_cls

    def forward_threshold(self, x, threshold):
        feature1 = F.relu(self.bn1(self.conv1(x)))
        feature2 = self.layer1(feature1)
        feature3 = self.layer2(feature2)
        feature4 = self.layer3(feature3)
        feature5 = self.layer4(feature4)
        feature5 = self.avgpool(feature5)
        feature = feature5.clip(max=threshold)
        feature = feature.view(feature.size(0), -1)
        logits_cls = self.fc(feature)
        return logits_cls

    def intermediate_forward(self, x, layer_index):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        if layer_index == 1:
            return out
        out = self.layer2(out)
        if layer_index == 2:
            return out
        out = self.layer3(out)
        if layer_index == 3:
            return out
        out = self.layer4(out)
        if layer_index == 4:
            return out
        raise ValueError

    def get_fc(self):
        fc = self.fc
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()

    def get_fc_layer(self):
        return self.fc


def preactresnet18_32x32(num_classes=10):
    return PreActResNet18_32x32(num_classes=num_classes)
