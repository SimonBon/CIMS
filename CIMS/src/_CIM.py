import torch
from torch import nn
from typing import Literal
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
import torch

class SE(nn.Module):

    def __init__(self, in_channels, width_per_channel, reduction=4):
        super().__init__()
        
        self.in_channels = in_channels 
        self.width_per_channel = width_per_channel 
        self.total_channels = in_channels * width_per_channel
        
        per_marker_hidden = max(1, width_per_channel // reduction)
        
        self.fc1 = nn.Conv2d(self.total_channels, 
                             in_channels * per_marker_hidden, 
                             kernel_size=1, 
                             groups=self.in_channels, 
                             bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.fc2 = nn.Conv2d(in_channels * per_marker_hidden, 
                             self.total_channels, 
                             kernel_size=1, 
                             groups=self.in_channels, 
                             bias=True)
        
    def forward(self, x):
        
        y = F.adaptive_avg_pool2d(x, 1)  
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = torch.sigmoid(y)
        
        return x * y

class Block(nn.Module):
    def __init__(self, total_channels, in_channels, width_per_channel, kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv2d(
            total_channels, 
            total_channels, 
            kernel_size, 
            padding=kernel_size//2,
            groups=total_channels, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(total_channels)
        
        self.se = SE(in_channels, width_per_channel, reduction=4)
        
        self.conv2 = nn.Conv2d(
            total_channels,
            total_channels, 
            kernel_size=1,
            groups=total_channels, 
            bias=False
        )
        
        self.bn2 = nn.BatchNorm2d(total_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.se(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        return out

class CIM(nn.Module):
    
    def __init__(self, in_channels, width_per_channel=16, layers=[2, 2, 2, 2], late_fusion=False):
        super().__init__()
        
        self.in_channels = in_channels
        self.width_per_channel = width_per_channel
        self.total_channels = in_channels * width_per_channel
        
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                self.total_channels, 
                kernel_size=3, 
                padding=1, 
                groups=in_channels, 
                bias=False
            ),
            nn.BatchNorm2d(self.total_channels),
            nn.ReLU(inplace=True)
        )
        
        blocks = []
        for layer in layers:
            blocks.append(self._make_layer(layer))
        
        if late_fusion:
            
            self.late_fusion = nn.Conv2d(
                self.total_channels,
                self.total_channels,
                kernel_size=1,
                padding=0,
                groups=1
            )
            
        else:
            
            self.late_fusion = None
            
        self.blocks = nn.Sequential(*blocks)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, blocks):
        layers = []
        
        for _ in range(blocks):
            layers.append(Block(
                total_channels=self.total_channels, 
                in_channels=self.in_channels,
                width_per_channel=self.width_per_channel
            ))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.stem(x)
        x = self.blocks(x)
        
        x = self.avgpool(x)
        if self.late_fusion is not None:
            x = self.late_fusion(x)
        
        return x
    
class CIMClassifier(CIM):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)

        n_feats = self.blocks[-1][-1].conv2.out_channels 
             
        self.classifier = nn.Sequential(
            nn.Linear(n_feats, n_feats // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(n_feats // 2, num_classes)
        )

    def forward(self, x):
        
        x = self.stem(x)
        x = self.blocks(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return self.classifier(x)
    
    
    
    
    
    
    
    
    
    
    
    
    
