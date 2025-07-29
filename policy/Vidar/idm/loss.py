import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveLoss(nn.Module):
    def __init__(self, target_precision, mean, std):
        super().__init__()
        # 注册缓冲区保存不可训练参数
        self.register_buffer('precision', target_precision.clone().detach())
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.register_buffer('ema', target_precision.clone().detach())    # EMA跟踪各维度MAE
        self.register_buffer('_current_mae', None)  # 临时存储当前batch的MAE
        
    def forward(self, y_pred_norm, y_true_norm):
        # denormalize
        mean = self.mean.to(y_pred_norm.device)
        std = self.std.to(y_pred_norm.device)
        precision = self.precision.to(y_pred_norm.device)
        
        y_pred = y_pred_norm * std + mean
        y_true = y_true_norm * std + mean
        
        abs_error = torch.abs(y_pred - y_true)  # [batch_size, 14]
        current_mae = torch.mean(abs_error, dim=0)  # [14]
        
        # 存储当前MAE（不参与梯度计算）
        self._current_mae = current_mae.detach().clone().cpu()
        
        # 动态权重计算（核心逻辑）
        # unmet_mask = (current_mae > precision).float()  # [14]
        unmet_mask = (self.ema > self.precision).float().to(y_pred_norm.device)  # 使用EMA代替current_mae
        # weights = (1.0 / (precision ** 2 + 1e-8)) * (1.0 + 2.0 * unmet_mask)  # [14]
        weights = (1.0 / (precision + 1e-8)) * unmet_mask  # [14]
        # TODO: 不用precision**2, 如果满足了精度就直接没有weight
        
        # 分离高精度和低精度维度
        # 高精度: [0:6], [7:13]
        # 低精度: [6:7], [13:14]
        high_precision_indices = torch.cat([torch.arange(0, 6), torch.arange(7, 13)]).to(y_pred.device)
        low_precision_indices = torch.tensor([6, 13]).to(y_pred.device)
        
        # MSE loss for high precision joints
        mse_part = (y_pred[:, high_precision_indices] - y_true[:, high_precision_indices]) ** 2
        
        # Smooth L1 loss for low precision joints
        l1_part = F.smooth_l1_loss(
            y_pred[:, low_precision_indices],
            y_true[:, low_precision_indices],
            reduction='none'
        )
        
        # 加权损失组合
        loss = 0.7 * torch.mean(weights[high_precision_indices] * mse_part) + \
               0.3 * torch.mean(weights[low_precision_indices] * l1_part)
        
        return loss

    def update_ema(self):
        """ 在训练循环中每batch调用此方法更新EMA """
        if self._current_mae is None:
            return
        
        # 动态调整EMA系数：未达标维度用快速更新（alpha=0.2），达标维度慢速更新（alpha=0.05）
        alpha = torch.where(
            self._current_mae > self.precision,
            torch.tensor(0.2, device=self.precision.device),
            torch.tensor(0.05, device=self.precision.device)
        )
        
        # EMA更新公式
        self.ema = (1 - alpha) * self.ema + alpha * self._current_mae


class NTKInspiredAdaptiveLoss(nn.Module):
    def __init__(self, target_precision, data_range, mean, std, 
                 ema_alpha=0.3, temp=1.0, grad_balance=True):
        super().__init__()
        # 注册缓冲区
        self.register_buffer('target_precision', target_precision)
        self.register_buffer('data_range', data_range)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        
        # 动态参数
        self.ema_alpha = ema_alpha
        self.temp = temp  # 权重锐度调节
        self.grad_balance = grad_balance
        
        # 状态跟踪
        self.register_buffer('ema_mae', torch.zeros_like(target_precision))
        self.register_buffer('initialized', torch.tensor(False))
        
        # 可学习权重（梯度平衡）
        if grad_balance:
            self.task_weights = nn.Parameter(torch.ones(len(target_precision)))
        
    def forward(self, y_pred_norm, y_true_norm):
        # 反归一化
        mean = self.mean.to(y_pred_norm.device)
        std = self.std.to(y_pred_norm.device)
        target_precision = self.target_precision.to(y_pred_norm.device)
        data_range = self.data_range.to(y_pred_norm.device)
        ema_mae = self.ema_mae.to(y_pred_norm.device)
        initialized = self.initialized.to(y_pred_norm.device)
        
        y_pred = y_pred_norm * std + mean
        y_true = y_true_norm * std + mean
        
        # 计算相对误差（数据范围归一化）
        abs_error = torch.abs(y_pred - y_true) / data_range  # [B,14]
        batch_mae = torch.mean(abs_error, dim=0)  # [14]
        
        # 更新EMA
        if not initialized:
            # 确保在同一设备上进行操作
            self.ema_mae.copy_(batch_mae.detach())
            self.initialized.fill_(True)
        else:
            # 所有计算都在当前设备上进行，不要使用cpu()
            new_ema = (1 - self.ema_alpha) * ema_mae + self.ema_alpha * batch_mae.detach()
            self.ema_mae.copy_(new_ema)
        
        # 核心权重计算（参考NTK平衡思想）
        precision_ratio = self.ema_mae.to(y_pred_norm.device) / (target_precision + 1e-8)
        precision_ratio = torch.clamp(precision_ratio, max=20.0)  # 防止指数爆炸
        weights = torch.exp(self.temp * (precision_ratio - 1))  # 指数放大未达标项
        
        # 梯度平衡因子（可学习权重）
        if self.grad_balance:
            weights = weights * F.softplus(self.task_weights.to(y_pred_norm.device))
        
        # 归一化权重（保持平均权重为1）
        weights = weights / weights.mean().detach()
        
        # 加权MSE损失
        weighted_mse = torch.mean(weights * (abs_error ** 2), dim=1)
        loss = torch.mean(weighted_mse)
        
        return loss

    def get_current_metrics(self):
        """ 返回各维度监控指标 """
        return {
            'ema_mae': self.ema_mae.cpu().numpy(),
            'target_precision': self.target_precision.cpu().numpy(),
            'precision_ratio': (self.ema_mae/self.target_precision).cpu().numpy()
        }
    

class WeightedSmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0, learning_dim=None):
        super().__init__()
        weights = torch.ones(14)  # 基础权重为1
        wrist_indices = [4, 11]  # weighted joints
        weights[wrist_indices] = 2.0  # 手腕关节权重为2
        self.register_buffer('joint_weights', weights)
        self.beta = beta
        
        # Create learning dimension mask
        self.dim_mask = torch.zeros(14, dtype=torch.bool)
        if learning_dim is not None:
            for dim in learning_dim:
                if 0 <= dim < 14:
                    self.dim_mask[dim] = True
        else:
            self.dim_mask.fill_(True)  # Default: learn all dimensions
        self.register_buffer('learning_mask', self.dim_mask)

    def forward(self, pred, target):
        # 计算SmoothL1Loss
        diff = torch.abs(pred - target)
        smooth_l1_loss = torch.where(diff < self.beta,
                                   0.5 * diff * diff / self.beta,
                                   diff - 0.5 * self.beta)
        
        # 应用关节权重并且只针对需要学习的维度计算损失
        learning_mask = self.learning_mask.to(pred.device)
        weights = self.joint_weights.view(1, -1).to(pred.device)
        masked_weights = weights * learning_mask.float()
        weighted_loss = smooth_l1_loss * masked_weights.view(1, -1)
        
        # Only average over dimensions we're learning
        active_dims = learning_mask.sum()
        if active_dims > 0:
            return weighted_loss.sum() / (pred.size(0) * active_dims)
        else:
            return torch.tensor(0.0, device=pred.device)



class WeightedL2Loss(nn.Module):
    def __init__(self):
        super().__init__()
        weights = torch.ones(14)  # 基础权重为1
        wrist_indices = [4, 11]  # weighted joints
        weights[wrist_indices] = 2.0  # 手腕关节权重为2
        self.register_buffer('joint_weights', weights)

    def forward(self, pred, target):
        # Calculate L2 loss (squared error)
        squared_diff = (pred - target) ** 2
        
        # Apply joint weights
        weighted_loss = squared_diff * self.joint_weights.view(1, -1).to(pred.device)
        
        return weighted_loss.mean()
    

class DynamicWeight(nn.Module):
    """ 动态权重, https://pmc.ncbi.nlm.nih.gov/articles/PMC11504533/ """
    def __init__(self, num_tasks=3):
        self.weights = nn.Parameter(torch.ones(num_tasks))
        
    def forward(self, losses):
        return torch.sum(F.softmax(self.weights,0) * torch.stack(losses))
