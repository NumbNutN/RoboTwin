import torch
import torch.nn as nn
import torch.functional as F

class RobotContrastivePolicy(nn.Module):
    def __init__(self, 
                 joint_dim=7,
                 traj_len=10,
                 embed_dim=512,
                 dino_model_name='dinov2_vits14'):
        super().__init__()

        # ---------------------------------
        # Visual Tower
        # ---------------------------------
        self.visual_backbone = torch.hub.load('facebookresearch/dinov2',dino_model_name)

        # 冻结视觉骨干参数 (Very important! We only train mapping layer)
        for param in self.visual_backbone.parameters():
            param.requires_grad = False

        visual_feat_dim = 384

        # 视觉投影头：将Dino特征映射到共享嵌入空间
        self.visual_projector = nn.Sequential(
            nn.Linear(visual_feat_dim,512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

        # ----------------------------------
        # Trajectory Tower
        # ----------------------------------
        # input shape: [Batch, Traj_Len, Joint_Dim] -> Flatten -> MLP

        # TODO: Can use Transformer/1D-CNN 时序, MLP here for demonstrating
        input_dim = traj_len * joint_dim

        self.traj_encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512,embed_dim)
        )

        # TODO: 这里的 Log Temperature 用于缩放对比学习的 logits
        self.logit_scale = nn.Parameter(torch.one([]) * 2.6592) # init to e^2.6592 = 14

    def encode_image(self, images):
        with torch.no_grad():
            # DINOv2 通常需要特定的transform (resize, normalize)
            features = self.visual_backbone(images) # [Batch, 384]

        embeddings = self.visual_projector(features)
        return F.normalize(embeddings, dim=-1) # 归一化非常关键
    
    def encode_trajectory(self, trajectories):
        B, T, D = trajectories.shape
        flat_traj = trajectories.view(B,-1)

        embeddings = self.traj_encoder(flat_traj)
        return F.normalize(embeddings, -1)
    
    def forward(self, images, traj):
        # 训练时同时通过两个塔
        img_embeds = self.encode_image(images)
        traj_embeds = self.encode_trajectory(traj)
        return img_embeds, traj_embeds