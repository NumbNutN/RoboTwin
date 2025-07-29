import torch
import torch.nn as nn
from transformers import Dinov2WithRegistersModel
import torch.nn.functional as F

from .direction_aware import *
from .resnet import *
from .heatmap import *
from .unet import *


class IDM(nn.Module):

    def __init__(self, model_name, *args, **kwargs):
        super(IDM, self).__init__()
        match model_name:
            case "dino":
                self.model = DINO(*args, **kwargs)
            case "direction_aware":
                self.model = DirectionAwareDINO(*args, **kwargs)
            case "mask":
                self.model = Mask(*args, **kwargs)
            case "unet":
                self.model = Unet(*args, **kwargs)
            case "resnet":
                self.model = ResNet(*args, **kwargs)
            case "resnet50":
                self.model = ResNet50Regressor(*args, **kwargs)
            case "heatmap":
                self.model = HeatmapDINO(*args, **kwargs)
            case "direction_aware_with_split":
                self.model = DirectionAwareDINOWithSplitLines(*args, **kwargs)
            case "resnet_with_split":
                self.model = ResnetWithSplitLines(*args, **kwargs)
            case "dino_with_split":
                self.model = DINOWithSplitLines(*args, **kwargs)
            case _:
                raise ValueError(f"Unsupported model name: {model_name}")
        train_mean = torch.tensor([-0.26866713, 0.83559588, 0.69520934, -0.29099351, 0.18849116, -0.01014598, 1.41953145, 0.35073715, 1.05651613, 0.8930193, -0.37493264, -0.18510782, -0.0272574, 1.35274259])
        train_std = torch.tensor([0.25945241, 0.65903812, 0.52147207, 0.42150272, 0.32029947, 0.28452226, 1.78270006, 0.29091741, 0.67675932, 0.58250554, 0.42399049, 0.28697442, 0.31100304, 1.67651926])
        self.register_buffer("train_mean", train_mean)
        self.register_buffer("train_std", train_std)

    def normalize(self, x):
        x = (x - self.train_mean) / self.train_std
        return x

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        if isinstance(output, tuple):
            return output[0] * self.train_std + self.train_mean, *output[1:]
        else:
            return output * self.train_std + self.train_mean


class Unet(nn.Module):
    def __init__(self, output_dim: int = 14, *args, **kwargs):

        super().__init__()
        self.output_dim = output_dim

        # Initialize DINO v2 model
        self.mask_model = UNet(3, 3)
        self.resnet_model = ResNet(14, 3)

        # Print number of parameters
        print(f"output_dim: {output_dim}, parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, images, *args, **kwargs):
        outputs = self.resnet_model(torch.tanh(self.mask_model(images)))
        return outputs


class Mask(nn.Module):
    def __init__(self, output_dim: int = 14, *args, **kwargs):

        super().__init__()
        self.output_dim = output_dim

        # Initialize DINO v2 model
        self.mask_model = UNet(3, 1)
        self.resnet_model = ResNet(14, 3)

        # Print number of parameters
        print(f"output_dim: {output_dim}, parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, images, return_mask=False, *args, **kwargs):
        mask = (1 + torch.tanh(self.mask_model(images))) / 2
        mask_hard = torch.where(mask < 0.5, 0.0, 1.0)
        inputs = images * ((mask_hard - mask).detach() + mask)
        outputs = self.resnet_model(inputs)
        if return_mask:
            return outputs, mask
        else:
            return outputs


class DirectionAwareDINO(nn.Module):
    def __init__(self, dinov2_name: str = "facebook/dinov2-with-registers-base", freeze_dinov2 = False, output_dim: int = 14):
        
        super().__init__()
        self.output_dim = output_dim
        
        # Initialize DINO v2 model
        self.dino_model = Dinov2WithRegistersModel.from_pretrained(dinov2_name)

        if freeze_dinov2:
            for param in self.dino_model.parameters():
                param.requires_grad_(False)

        # Get model configuration parameters
        hidden_size = self.dino_model.config.hidden_size  # 768
        patch_size = self.dino_model.config.patch_size  # 14
        self.dino_wh = 518
        self.hw_size = self.dino_wh // patch_size  # 37 for 518/14
        
        # Initialize model components
        # self.orientation_head = OrientationAwareHead(hidden_size)
        # self.kinematics_layer = KinematicsLayer(num_joints=output_dim)
        angle_num = 4
        self._angle_num = angle_num
        self.direction_decoder = DirectionAwareDecoder(hidden_size, self._angle_num)
        
        # Final regressor
        self.regressor = nn.Sequential(
            nn.Linear(256*self._angle_num, 512),  # 增加角度特征的通道
            nn.GELU(),
            nn.Linear(512, output_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Print number of parameters
        print(f"dinov2_name: {dinov2_name}, freeze_dinov2: {freeze_dinov2}, output_dim: {output_dim}, parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, images, *args, **kwargs):
        hidden_size = self.dino_model.config.hidden_size  # 768
        # Process input images
        inputs_dinov2 = images  # [B, C, H, W]
            
        # Get DINO v2 embeddings
        outputs = self.dino_model(inputs_dinov2)
        last_hidden_state = outputs.last_hidden_state  # [B, 1+num_register+N, hidden_size]
        
        # Extract patch embeddings (skip CLS and register tokens)
        num_register_tokens = self.dino_model.config.num_register_tokens  # Usually 4
        patch_embeddings = last_hidden_state[:, num_register_tokens + 1:, :]  # [B, N, hidden_size]
        patch_embeddings = self.layer_norm(patch_embeddings)  # [B, N, hidden_size]
        
        # Get joint angles from orientation-aware head
        # angles = self.orientation_head(patch_embeddings, self.dino_model.config.patch_size, self.hw_size)  # [B, output_dim]
        
        # Apply direction-aware decoder for additional features
        direction_features = self.direction_decoder(
            patch_embeddings.view(-1, self.hw_size, self.hw_size, hidden_size).permute(0,3,1,2)  # [B, hidden_size, hw_size, hw_size]
        )  # [B, 1024]
        
        predictions = self.regressor(direction_features)  # [B, output_dim]

        return predictions


class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, tokenW=32, tokenH=32, num_labels=1, output_size=256, relative=False):
        super(LinearClassifier, self).__init__()
        # the original num_labels is 2
        self.in_channels = in_channels  # 768
        self.width = tokenW  # 518/14 = 37
        self.height = tokenH  # 518/14 = 37
        self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1,1))
        self.fc = nn.Linear(num_labels*tokenW*tokenH, output_size)  # 2*37*37 = 2738

    def forward(self, embeddings, *args, **kwargs):
        # embeddings shape is [B, 1369, 768], 1369 = 37 * 37
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)  # [B, 37, 37, 768]
        embeddings = embeddings.permute(0,3,1,2).contiguous()  # [B, 768, 37, 37]
        
        output = self.classifier(embeddings)  # [B, 2, 37, 37]
        output = output.reshape(output.size(0), -1)  # [B, 2*37*37]
        output = self.fc(output)  # [B, 256]

        return output


class DINO(nn.Module):
    def __init__(self, dinov2_name: str = "facebook/dinov2-with-registers-base", freeze_dinov2 = False, output_dim: int = 14):
        
        super().__init__()
        self.dino_model = Dinov2WithRegistersModel.from_pretrained(dinov2_name) 
        self.output_dim = output_dim

        if freeze_dinov2:
            for param in self.dino_model.parameters():
                param.requires_grad_(False)

        hidden_size = self.dino_model.config.hidden_size
        num_labels = self.dino_model.config.num_labels
        self.dino_wh = 518
        patch_size = self.dino_model.config.patch_size  # 14

        # Original implementation with shared layers
        self.linear = LinearClassifier(hidden_size, self.dino_wh // patch_size, self.dino_wh // patch_size, num_labels, 256, False)
        self.regressor = nn.Sequential(
            nn.GELU(),
            # nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )

        self.layer_norm = nn.LayerNorm(self.dino_model.config.hidden_size)
        print(f"dinov2_name: {dinov2_name}, freeze_dinov2: {freeze_dinov2}, output_dim: {output_dim}, parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")
        
    def forward(self, images, *args, **kwargs):
        outputs = self.dino_model(images) 
        last_hidden_state = outputs.last_hidden_state  # [B, 1374, 768]
        # feat_layers = [outputs.hidden_states[i] for i in [4, 8, 12]]  # 选择中间层
        # # 特征融合（示例）：
        # fused_feat = torch.cat([
        #     F.interpolate(feat, size=(37,37), mode='bilinear') 
        #     for feat in feat_layers
        # ], dim=1)  # [B, 3 * 768, 37, 37]

        # For Dinov2
        # patch_embeddings = last_hidden_state[:,1:,:]

        # For Dinov2WithRegistersModel, we need to handle the register tokens
        # The first token is CLS, followed by register tokens, then patch tokens
        num_register_tokens = self.dino_model.config.num_register_tokens  # Usually 4 for dinov2-with-registers
        patch_embeddings = last_hidden_state[:, num_register_tokens + 1:, :]  # Skip CLS and register tokens
        patch_embeddings = self.layer_norm(patch_embeddings)  # [B, 1369, 768]
        
        #     Original implementation
        outputs = self.linear(patch_embeddings)
        predictions = self.regressor(outputs)
        return predictions


class HeatmapDINO(nn.Module):
    """
    # use example of HeatmapRegressionHead and DenseDecoder
        patch_size = 14
        tokenW = self.dino_wh // patch_size  # 37 when dino_wh=518
        tokenH = self.dino_wh // patch_size
        
        self.decoder = DenseDecoder(dinov2_hidden_size=768, tokenH=tokenH, tokenW=tokenW)
        self.heatmap_head = HeatmapRegressionHead(
            dinov2_hidden_size=768, 
            num_joints=14,
            tokenH=tokenH,
            tokenW=tokenW
        )
        
        # 权重融合层
        self.fusion = nn.Linear(num_joints*3,14)  # 14*3=42

    def forward(self, x):
        dino_out = self.dino_model(x)
        patch_emb = ...  # 提取patch embeddings
        
        out1 = self.decoder(patch_emb)  # [B, 14]
        out2 = self.heatmap_head(patch_emb)  # [B, 28]
        
        # 特征拼接后融合
        combined = torch.cat([out1, out2], dim=-1)
        return self.fusion(combined)
    """
    def __init__(self, dinov2_name: str = "facebook/dinov2-base", freeze_dinov2 = False, output_dim: int = 14):
        
        super().__init__()
        # 初始化DINOv2模型
        self.dino_model = Dinov2WithRegistersModel.from_pretrained(dinov2_name)
        self.output_dim = output_dim
        
        # 冻结DINOv2参数
        if freeze_dinov2:
            for param in self.dino_model.parameters():
                param.requires_grad_(False)
        
        # 从配置获取模型参数
        hidden_size = self.dino_model.config.hidden_size  # 768
        self.dino_wh = 518
        patch_size = self.dino_model.config.patch_size  # 14
        self.tokenH = self.tokenW = self.dino_wh // patch_size  # 37 when dino_wh=518
        
        # 初始化解码器模块
        self.decoder = DenseDecoder(
            dinov2_hidden_size=hidden_size,
            tokenH=self.tokenH,
            tokenW=self.tokenW
        )
        self.heatmap_head = HeatmapRegressionHead(
            dinov2_hidden_size=hidden_size,
            num_joints=output_dim,
            tokenH=self.tokenH,
            tokenW=self.tokenW
        )
        
        # 特征融合层
        self.fusion = nn.Linear(output_dim*3, output_dim)  # 14*3=42 -> 14
        
        print(f"dinov2_name: {dinov2_name}, freeze_dinov2: {freeze_dinov2}, output_dim: {output_dim}, parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, images, *args, **kwargs):
        # 输入形状: [B, 3, H, W]
        B = images.size(0)
        
        # DINOv2特征提取
        outputs = self.dino_model(images)
        last_hidden_state = outputs.last_hidden_state  # [B, num_tokens, 768]
        
        # 处理register tokens
        num_register_tokens = self.dino_model.config.num_register_tokens  # 通常为4
        patch_embeddings = last_hidden_state[:, num_register_tokens + 1:, :]  # [B, 1369, 768]
        
        # 通过两个解码分支
        dense_out = self.decoder(patch_embeddings)  # [B, 14]
        heatmap_out = self.heatmap_head(patch_embeddings)  # [B, 28]
        
        # 特征拼接与融合
        combined = torch.cat([dense_out, heatmap_out], dim=-1)  # [B, 42]
        return self.fusion(combined)  # [B, 14]


class DirectionAwareDINOWithSplitLines(nn.Module):
    def __init__(self, dinov2_name: str = "facebook/dinov2-with-registers-base", freeze_dinov2=False, output_dim: int = 14):
        super().__init__()
        
        # 创建四个不同输出的模型
        self.region_models = nn.ModuleList([
            # 左上区域模型，输出6维
            self._build_region_model(dinov2_name, freeze_dinov2, output_dim=6),
            # 左下区域模型，输出1维
            # self._build_region_model(dinov2_name, freeze_dinov2, output_dim=1),
            DINO(
                dinov2_name=dinov2_name,
                freeze_dinov2=freeze_dinov2,
                output_dim=1,
            ),
            # 右上区域模型，输出6维
            self._build_region_model(dinov2_name, freeze_dinov2, output_dim=6),
            # 右下区域模型，输出1维
            # self._build_region_model(dinov2_name, freeze_dinov2, output_dim=1)
            DINO(
                dinov2_name=dinov2_name,
                freeze_dinov2=freeze_dinov2,
                output_dim=1,
            ),
        ])
    
    def _build_region_model(self, dinov2_name, freeze_dinov2, output_dim):
        """构建区域专用模型"""
        return DirectionAwareDINO(
                dinov2_name=dinov2_name,
                freeze_dinov2=freeze_dinov2,
                output_dim=output_dim
            )

    def forward(self, region_images, *args, **kwargs):
        # 输入是四个区域的图像列表, region_images: [4, B, 3, H, W]
        # TODO: share the backbone
        
        # 拼接各个区域的输出
        final_output = torch.cat([
            self.region_models[0](region_images[0]),  # 0-6维
            self.region_models[1](region_images[1]),  # 6-7维
            self.region_models[2](region_images[2]),  # 7-13维
            self.region_models[3](region_images[3])   # 13-14维
        ], dim=1)
        
        return final_output  # [B, 14]

class DINOWithSplitLines(nn.Module):
    def __init__(self, dinov2_name: str = "facebook/dinov2-with-registers-base", freeze_dinov2=False, output_dim: int = 14):
        super().__init__()
        
        # 创建四个不同输出的模型
        self.region_models = nn.ModuleList([
            # 左上区域模型，输出6维
            self._build_region_model(dinov2_name, freeze_dinov2, output_dim=6),
            # 左下区域模型，输出1维
            self._build_region_model(dinov2_name, freeze_dinov2, output_dim=1),
            # 右上区域模型，输出6维
            self._build_region_model(dinov2_name, freeze_dinov2, output_dim=6),
            # 右下区域模型，输出1维
            self._build_region_model(dinov2_name, freeze_dinov2, output_dim=1)
        ])
    
    def _build_region_model(self, dinov2_name, freeze_dinov2, output_dim):
        """构建区域专用模型"""
        return DINO(
                dinov2_name=dinov2_name,
                freeze_dinov2=freeze_dinov2,
                output_dim=output_dim
            )

    def forward(self, region_images, *args, **kwargs):
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
