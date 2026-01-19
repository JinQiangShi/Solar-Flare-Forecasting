from torch import flatten, Tensor
import torch.nn as nn
from torchvision.models import inception_v3, resnet34

from warnings import filterwarnings
filterwarnings("ignore")

class AlexNet(nn.Module):
    def __init__(self, input_size=(512,512), num_classes = 4):
        super().__init__()

        # 卷积神经网络的卷积部分
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(192), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2))

        # 卷积神经网络的全局池化层
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # 重写全连接层（加入BatchNorm）
        self.classifier = nn.Sequential(
            nn.Dropout(), 
            nn.Linear(9216, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True), 
            nn.Dropout(), 
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes))
        
    def forward(self, x:Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = flatten(x, 1)
        x = self.classifier(x)
        return x
    
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class VGG(nn.Module):
    def __init__(self, num_classes = 4):
        super().__init__()
        
        self.features = nn.Sequential(
            BasicConv(3, 64, kernel_size=3, stride=1, padding=1),
            BasicConv(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BasicConv(64, 128, kernel_size=3, stride=1, padding=1),
            BasicConv(128, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BasicConv(128, 256, kernel_size=3, stride=1, padding=1),
            BasicConv(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BasicConv(256, 512, kernel_size=3, stride=1, padding=1),
            BasicConv(512, 512, kernel_size=3, stride=1, padding=1), 
            BasicConv(512, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            BasicConv(512, 512, kernel_size=3, stride=1, padding=1),
            BasicConv(512, 512, kernel_size=3, stride=1, padding=1),
            BasicConv(512, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # 重写全连接层（加入BatchNorm）
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096), 
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True), 
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True), 
            nn.Dropout(),
            nn.Linear(4096, num_classes))

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = flatten(x, 1)
        x = self.classifier(x)
        return x

def Inception3(num_classes=4):
    model = inception_v3()
    model.fc = nn.Linear(2048, num_classes)
    return model

def ResNet(num_classes=4):
    model = resnet34()
    model.fc = nn.Linear(512, num_classes)
    return model