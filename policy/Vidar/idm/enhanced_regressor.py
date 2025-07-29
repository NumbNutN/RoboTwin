import torch
import torch.nn as nn
from transformers import Dinov2WithRegistersModel
import torch.nn.functional as F


class SpatialFeatureExtractor(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256):
        super().__init__()
        # 使用膨胀卷积捕捉多尺度上下文
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, padding=6, dilation=6),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=3, dilation=3),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1)
        )
        # 坐标编码层
        self.coord_conv = CoordConv(hidden_dim, hidden_dim)
        
    def forward(self, x):
        # x shape: [B, 37, 37, 768]
        x = x.permute(0,3,1,2)  # [B,768,H,W]
        x = self.conv_block(x)  # [B,256,H,W]
        x = self.coord_conv(x)  # 注入空间坐标信息
        return x  # [B,256,37,37]


class CoordConv(nn.Module):
    """坐标特征增强模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels+2, out_channels, 3, padding=1)
        
    def forward(self, x):
        batch, _, h, w = x.shape
        # 生成坐标网格
        x_coord = torch.linspace(-1, 1, w).repeat(h,1)
        y_coord = torch.linspace(-1, 1, h).repeat(w,1).t()
        grid = torch.stack([x_coord, y_coord], dim=0).unsqueeze(0).repeat(batch,1,1,1)
        grid = grid.to(x.device)
        # 拼接坐标信息
        x = torch.cat([x, grid], dim=1)
        return self.conv(x)
    

class OrientationAwareBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 方向卷积核组
        self.directional_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels//4, 3, padding=1),  # out: [B, 64, 37, 37]
            nn.Conv2d(in_channels, in_channels//4, 3, padding=2, dilation=2),  # out: [B, 64, 37, 37]
            nn.Conv2d(in_channels, in_channels//4, (1,3), padding=(0,1)),  # out: [B, 64, 37, 37]
            nn.Conv2d(in_channels, in_channels//4, (3,1), padding=(1,0))  # out: [B, 64, 37, 37]
        ])
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, spatial_feat):
        # spatial_feat: [B, 256, 37, 37]
        features = [conv(spatial_feat) for conv in self.directional_convs]  # [B, 64, 37, 37] for each feature
        combined = torch.cat(features, dim=1)  # [B, 256, 37, 37]
        attn = self.attention(spatial_feat)  # [B, 1, 37, 37]
        return combined * attn  # [B, 256, 37, 37]


class EnhancedRegressor(nn.Module):
    def __init__(self, 
                 use_depth = False,
                 dinov2_name: str = "facebook/dinov2-base",
                 freeze_dinov2 = False,
                 output_dim: int = 14):
        super().__init__()
        
        # 初始化DINOv2模型
        self.dino_model = Dinov2WithRegistersModel.from_pretrained(dinov2_name)
        self.output_dim = output_dim
        self.use_depth = use_depth
        
        # 从模型配置获取参数
        hidden_size = self.dino_model.config.hidden_size  # 768
        patch_size = self.dino_model.config.patch_size    # 14
        
        
        # 动态计算特征图尺寸
        self.dino_wh = 518
        self.token_size = self.dino_wh // patch_size  # 37 when dino_wh=518
        
        if freeze_dinov2:
            for param in self.dino_model.parameters():
                param.requires_grad_(False)

        # 特征增强模块
        self.spatial_extractor = SpatialFeatureExtractor(
            in_dim=hidden_size, 
            hidden_dim=256
        )
        
        # 方向感知模块
        self.orientation_block = OrientationAwareBlock(256)
        
        # 多尺度融合
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(256*3, 512, 1),
            nn.GELU(),
            nn.Conv2d(512, 256, 3, padding=1)
        )
        
        # 关节预测分支
        self.joint_branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # [B, 768, 1, 1]
                nn.Flatten(),  # [B, 768]
                nn.Linear(768, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            ) for _ in range(output_dim)
        ])

    def forward(self, images):
        # images shape is [B, 3, 518, 518]

        # 特征提取
        outputs = self.dino_model(images) 
        num_register_tokens = self.dino_model.config.num_register_tokens  # 4
        patch_embeddings = outputs.last_hidden_state[:, num_register_tokens+1:, :]  # 跳过CLS和寄存器token, [B, 1369, 768]
        patch_embeddings = patch_embeddings.reshape(-1, self.token_size, self.token_size, 768)  # [B, 37, 37, 768]
        
        # 特征增强流程
        spatial_feat = self.spatial_extractor(patch_embeddings)  # [B, 256, 37, 37]  (token_size=37)
        orientation_feat = self.orientation_block(spatial_feat)  # [B, 256, 37, 37]
        
        # 特征融合（结合空间特征和方向特征）
        enhanced_feat = spatial_feat + orientation_feat  # 特征相加 [B, 256, 37, 37]
        
        # 多尺度特征融合（使用增强后的特征）
        down_feat = F.avg_pool2d(enhanced_feat, 3, stride=2, padding=1)  # [B, 256, 18, 18]
        up_feat = F.interpolate(enhanced_feat, scale_factor=2)  # [B, 256, 74, 74]
        fused_feat = torch.cat([
            enhanced_feat,  # 使用增强后的特征
            F.interpolate(down_feat, size=(self.token_size, self.token_size)),
            F.adaptive_avg_pool2d(up_feat, (self.token_size, self.token_size))
        ], dim=1)  # [B, 768, 37, 37]
        
        # 各关节独立预测
        predictions = [branch(fused_feat) for branch in self.joint_branches]  # [B, 1, 37, 37] for each branch
        return torch.cat(predictions, dim=1)  # [B, 14]
