import torch
import torch.nn as nn
import torch.nn.functional as F

class FastPrimaryCaps2D(nn.Module):

    def __init__(self, in_channels, num_capsules, capsule_dim):
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        # 卷积出 num_capsules * capsule_dim 通道
        self.conv = nn.Conv2d(in_channels, num_capsules * capsule_dim, kernel_size=3, padding=1)

    def forward(self, x):  # x: [B, C, H, W]
        B, C, H, W = x.size()
        out = self.conv(x)  # [B, num_capsules*capsule_dim, H, W]
        # 每个空间位置为一个 primary capsule
        out = out.view(B, self.num_capsules, self.capsule_dim, H * W)
        out = out.permute(0, 3, 1, 2).contiguous()  # [B, H*W, num_capsules, capsule_dim]
        out = out.view(B, -1, self.capsule_dim)        # [B, N_primary, capsule_dim]
        # squash 激活
        squared = (out**2).sum(dim=-1, keepdim=True)
        scale = squared / (1 + squared) / torch.sqrt(squared + 1e-8)
        return scale * out  # [B, N_primary, capsule_dim]

class FastCapsNetMulti(nn.Module):
    def __init__(self,
                 hsi_channels, sar_channels,
                 num_capsules=4, capsule_dim=8,
                 class_dim=16, num_classes=10):
        super().__init__()
        # Primary capsules
        self.hsi_primary = FastPrimaryCaps2D(hsi_channels, num_capsules, capsule_dim)
        self.sar_primary = FastPrimaryCaps2D(sar_channels, num_capsules, capsule_dim)
        # 合并后总 capsule 数
        self.routing_iters = 1
        # W 用于将 primary 转换到 class capsule
        # N_primary = (H*W*num_capsules)*2，临时在 forward 里扩展
        self.class_dim = class_dim
        self.num_classes = num_classes
        self.W = None  # 延迟初始化

    def forward(self, x_hsi, x_sar):
        B = x_hsi.size(0)
        # 1. Primary capsules
        caps_h = self.hsi_primary(x_hsi)  # [B, N1, D]
        caps_s = self.sar_primary(x_sar)  # [B, N2, D]
        caps = torch.cat([caps_h, caps_s], dim=1)  # [B, N, D]
        B, N, D = caps.size()
        # 2. 初始化 W
        if self.W is None or self.W.size(1) != N:
            # W: [1, N, num_classes, class_dim, D]
            self.W = nn.Parameter(torch.randn(1, N, self.num_classes, self.class_dim, D, device=caps.device))
        # 3. 计算 u_hat: [B, N, num_classes, class_dim]
        x = caps.unsqueeze(2).unsqueeze(-1)  # [B, N,1,D,1]
        W = self.W.expand(B, -1, -1, -1, -1)
        u_hat = torch.matmul(W, x).squeeze(-1)  # [B, N, num_classes, class_dim]
        # 4. vectorized routing (一次迭代)
        b = torch.zeros(B, N, self.num_classes, device=caps.device)
        c = F.softmax(b, dim=1)  # across primary capsules
        s = (c.unsqueeze(-1) * u_hat).sum(dim=1)  # [B, num_classes, class_dim]
        # squash
        squared = (s**2).sum(dim=-1, keepdim=True)
        v = squared / (1 + squared) / torch.sqrt(squared + 1e-8) * s  # [B, num_classes, class_dim]
        # 5. 分类得分
        logits = torch.norm(v, dim=-1)  # [B, num_classes]
        return logits

if __name__ == "__main__":
    # 测试
    model = FastCapsNetMulti(hsi_channels=30, sar_channels=1,
                             num_capsules=2, capsule_dim=8,
                             class_dim=16, num_classes=10)
    hsi = torch.randn(8, 30, 10, 10)
    sar = torch.randn(8, 1, 10, 10)
    out = model(hsi, sar)
    print("Output shape:", out.shape)  # [8,10]


