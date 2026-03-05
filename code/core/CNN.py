import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNSpectralSAR(nn.Module):
    def __init__(self,
                 hsi_channels: int,
                 sar_channels: int,
                 patch_size: int,
                 num_classes: int):
        """
        Multi-source CNN-based classifier for HSI+SAR patches.

        Args:
            hsi_channels (int): 输入 HSI 波段数，比如 30
            sar_channels (int): 输入 SAR 通道数，比如 4
            patch_size (int): 空间 patch 大小，比如 10
            num_classes (int): 分类数
        """
        super(CNNSpectralSAR, self).__init__()
        # HSI 分支：1D 卷积混合光谱，再空间 2D 卷积
        self.hsi_spectral = nn.Sequential(
            # 输入 [B, C_hsi, P*P]
            nn.Conv1d(hsi_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.hsi_spatial = nn.Sequential(
            # 输入 [B, 128, P, P]
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        # SAR 分支：经典 2D 卷积
        self.sar_branch = nn.Sequential(
            nn.Conv2d(sar_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # 融合与分类
        fused_channels = 256 + 64
        self.fusion = nn.Sequential(
            nn.Conv2d(fused_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_hsi: torch.Tensor, x_sar: torch.Tensor) -> torch.Tensor:
        B, C_h, P, _ = x_hsi.shape
        # HSI 分支：1D 卷积
        x = x_hsi.view(B, C_h, -1)      # [B, C_h, P*P]
        x = self.hsi_spectral(x)        # [B, 128, P*P]
        x = x.view(B, 128, P, P)        # [B, 128, P, P]
        x = self.hsi_spatial(x)         # [B, 256, P, P]

        # SAR 分支
        s = self.sar_branch(x_sar)      # [B, 64, P, P]

        # 融合
        f = torch.cat([x, s], dim=1)    # [B, 320, P, P]
        f = self.fusion(f)              # [B, 256, 1, 1]
        logits = self.classifier(f)     # [B, num_classes]
        return logits

if __name__ == "__main__":
    model = CNNSpectralSAR(hsi_channels=30, sar_channels=1, patch_size=10, num_classes=10)
    hsi = torch.randn(8, 30, 10, 10)
    sar = torch.randn(8, 1, 10, 10)
    out = model(hsi, sar)
    print("Output shape:", out.shape)  # -> [8, 10]
