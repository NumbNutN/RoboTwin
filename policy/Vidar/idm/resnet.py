import torch.nn as nn
import torch
from transformers import ResNetModel, ResNetConfig
from torchvision.transforms import Resize, ToTensor


class BottleNeckResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * 4),
        )

        self.shortcut = nn.Sequential()

        if in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, output_dim=14, input_channels=3, *args, **kwargs):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.conv2_x = self._make_layer(64, 3, 1)
        self.conv3_x = self._make_layer(128, 4, 2)
        self.conv4_x = self._make_layer(256, 6, 2)
        self.conv5_x = self._make_layer(512, 3, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, output_dim)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BottleNeckResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * 4

        return nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


class ResnetWithSplitLines(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        # 创建四个不同输出的模型
        self.region_models = nn.ModuleList([
            # 左上区域模型，输出6维
            self._build_region_model(output_dim=6),
            # 左下区域模型，输出1维
            self._build_region_model(output_dim=1),
            # 右上区域模型，输出6维
            self._build_region_model(output_dim=6),
            # 右下区域模型，输出1维
            self._build_region_model(output_dim=1)
        ])
    
    def _build_region_model(self, output_dim):
        """构建区域专用模型"""
        return ResNet50Regressor(output_dim)

    def forward(self, region_images):
        # 输入是四个区域的图像列表, region_images: [4, B, 3, H, W]
        
        outputs = []
        for i in range(4):
            # 每个模型处理对应的区域
            out = self.region_models[i](region_images[i])
            outputs.append(out)
        
        # 拼接各个区域的输出
        final_output = torch.cat([
            outputs[0],  # 0-6维
            outputs[1],  # 6-7维
            outputs[2],  # 7-13维
            outputs[3]   # 13-14维
        ], dim=1)
        
        return final_output  # [B, 14]


class ResNet50Regressor(nn.Module):
    def __init__(self, output_dim=14, *args, **kwargs):
        super(ResNet50Regressor, self).__init__()
        self.resnet = ResNetModel.from_pretrained("microsoft/resnet-50")
        self.regressor = nn.Linear(2048, output_dim)
        self.transform =  nn.Sequential(
            Resize((224, 224)),
        )

    def forward(self, pixel_values):
        pixel_values = self.transform(pixel_values)
        outputs = self.resnet(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        flattened_output = pooled_output.view(pooled_output.size(0), -1) 
        regression_output = self.regressor(flattened_output)
        return regression_output
