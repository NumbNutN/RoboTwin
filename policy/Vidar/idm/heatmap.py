import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseDecoder(nn.Module):
    def __init__(self, dinov2_hidden_size=768, tokenH=37, tokenW=37):
        super().__init__()
        self.aspp = ASPP(dinov2_hidden_size, 256)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.GELU()
        )
        self.tokenH = tokenH
        self.tokenW = tokenW
        self.attn = SpatialAttention()
        self.final_conv = nn.Conv2d(128,14,1)

    def forward(self, patch_embeddings):
        # 输入形状: [B,1369,768]
        B = patch_embeddings.size(0)
        x = patch_embeddings.view(B, self.tokenH, self.tokenW, 768).permute(0,3,1,2)  # [B,768,37,37]
        
        # ASPP多尺度特征
        x = self.aspp(x)  # [B,256,37,37]
        
        # 上采样
        x = self.up1(x)  # [B,128,74,74]
        
        # 空间注意力
        x = self.attn(x)  # [B,128,37,37]

        x = self.final_conv(x)  # [B,14,37,37]
        
        # 全局平均+最大池化融合
        x = 0.5*(F.adaptive_avg_pool2d(x,1) + F.adaptive_max_pool2d(x,1))  # [B,14,1,1]
        return x.view(B,14)

# ASPP模块实现
class ASPP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, 1)
        self.conv2 = nn.Conv2d(in_dim, out_dim, 3, padding=6, dilation=6)
        self.conv3 = nn.Conv2d(in_dim, out_dim, 3, padding=12, dilation=12)
        self.conv4 = nn.Conv2d(in_dim, out_dim, 3, padding=18, dilation=18)
        self.fusion = nn.Conv2d(out_dim*4, out_dim, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        return self.fusion(torch.cat([x1,x2,x3,x4], dim=1))

# 空间注意力模块
class SpatialAttention(nn.Module):
    def forward(self, x):
        # x: [B,128,74,74]
        # 分步处理H和W维度
        avg_h = torch.mean(x, dim=2, keepdim=True)  # 沿H维度平均 [B,C,1,W]
        avg_w = torch.mean(avg_h, dim=3, keepdim=True)  # 沿W维度平均 [B,C,1,1]
        
        max_h, _ = torch.max(x, dim=2, keepdim=True)  # 沿H维度取最大 [B,C,1,W]
        max_w, _ = torch.max(max_h, dim=3, keepdim=True)  # 沿W维度取最大 [B,C,1,1]
        
        att = torch.sigmoid(avg_w + max_w)  # 合并通道信息
        return x * att  # 广播相乘 [B, C, H, W]


class HeatmapRegressionHead(nn.Module):
    def __init__(self, dinov2_hidden_size=768, num_joints=14, tokenH=37, tokenW=37):
        super().__init__()
        self.dinov2_hidden_size = dinov2_hidden_size
        self.tokenH = tokenH
        self.tokenW = tokenW
        
        # 特征压缩层
        self.feature_adapter = nn.Sequential(
            nn.Conv2d(dinov2_hidden_size, 256, 1),
            nn.GELU()
        )
        
        # 热图预测头
        self.heatmap_head = nn.Conv2d(256, num_joints, 3, padding=1)
        
        # 坐标回归分支
        self.coord_regressor = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),  # [B,64,37,37]
            nn.MaxPool2d(2),  # [B,64,18,18]
            nn.Flatten(),  # [B,64*18*18=20736]
            nn.Linear(64*(tokenH//2)*(tokenW//2), 256),   # 假设输入分辨率37x37经过下采样, [B,256]
            nn.Linear(256, num_joints*2)  # [B,28]
        )

    def forward(self, patch_embeddings):
        # 输入形状: [B, 1369, 768]
        B = patch_embeddings.size(0)
        # 转换为2D特征图
        x = patch_embeddings.view(B, self.tokenH, self.tokenW, self.dinov2_hidden_size).permute(0,3,1,2)  # [B,768,37,37]
        
        # 特征压缩
        x = self.feature_adapter(x)  # [B,256,37,37]
        
        # 热图分支
        heatmaps = self.heatmap_head(x)  # [B,14,37,37]
        
        # 坐标积分计算
        heatmaps = torch.softmax(heatmaps.view(B,14,-1), dim=-1)  # 空间softmax
        coord_x = torch.linspace(0,1,37, device=x.device)
        coord_y = torch.linspace(0,1,37, device=x.device)
        grid_y, grid_x = torch.meshgrid(coord_y, coord_x, indexing='ij')
        
        pred_x = (heatmaps * grid_x.flatten()).sum(dim=-1)  # [B,14]
        pred_y = (heatmaps * grid_y.flatten()).sum(dim=-1)  # [B,14]
        
        # 坐标回归分支
        coords = self.coord_regressor(x)  # [B,28]
        
        # 最终融合（可根据实际情况调整权重）
        # final_coords = 0.7*pred_x + 0.3*coords
        # 假设coords前14维是x，后14维是y
        final_x = 0.7*pred_x + 0.3*coords[:, :14]
        final_y = 0.7*pred_y + 0.3*coords[:, 14:]
        final_coords = torch.stack([final_x, final_y], dim=-1).view(B, -1)  # [B, 28]

        return final_coords
