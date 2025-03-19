import torch
from torch import nn
from torch.nn.modules.module import Module
from torchvision.models.resnet import resnet50

class Resnet50Backbone(Module):
    def __init__(self, in_channels, out_channels, pretrained=False):
        super(Resnet50Backbone, self).__init__()
        self.bone = resnet50(pretrained=pretrained)
        self.bone.conv1 = nn.Conv2d(in_channels, self.bone.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bone.fc = nn.Linear(self.bone.fc.in_features, out_channels)

    def forward(self, x):
        x = self.bone(x)
        return x