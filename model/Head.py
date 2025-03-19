from torch import nn
from torch.nn.modules.module import Module

class Head(Module):
    def __init__(self,in_channels,num_classes):
        super(Head,self).__init__()
        self.head=nn.Sequential(
            nn.ReLU(inplace=False),
            nn.BatchNorm1d(in_channels),
            nn.Linear(in_channels,num_classes,bias=False),
        )
        
    def forward(self,x):
        return self.head(x)