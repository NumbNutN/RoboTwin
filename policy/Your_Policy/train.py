import torch
import torch.nn.functional as F

def train_step(model, optimizer, images, traj):
    """
    images: [Batch, C, H, W]
    trajectories: [Batch, Traj_Len, Joint_Dim]
    """
    model.train()
    optimizer.zero_grad()

    # 前向传播
    img_embeds, traj_embeds = model(images, traj)

    # 计算余弦相似度矩阵
    # logit_scale 是可学习的温度系数(Temperature)
    logit_scale = model.logit_scale.exp()
    logits = logit_scale * (img_embeds @ traj_embeds.t())

    # logits shape: [Batch, Batch]
    # 对角线元素 logits[i][i] 是正样本对 （当前图片 vs 对应的真实轨迹
    # 非对角线元素 logits[i][j] 是负样本对 (当前图片 vs Batch里其他人的轨迹)

    # 构建标签
    batch_size = images.shape[0]
    labels = torch.arange(batch_size).to(images.device)

    # 双向 Cross Entropy Loss (像CLIP一样)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(),labels)
    loss = (loss_i + loss_t) / 2

    loss.backward()
    optimizer.step()
    return loss.item()