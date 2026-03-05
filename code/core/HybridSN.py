import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridSNMulti(nn.Module):
    def __init__(self, hsi_channels, sar_channels, num_classes, patch_size):
        """
        HybridSN 扩展到多源输入：HSI + SAR

        Args:
            hsi_channels (int): HSI 降维后通道数（如 30）。
            sar_channels (int): SAR 通道数（如 4）。
            num_classes (int): 分类类别数。
            patch_size (int): 小 patch 的空间尺寸（如 10）。
        """
        super(HybridSNMulti, self).__init__()
        self.patch_size = patch_size

        # --- HSI 3D 卷积分支 ---
        # 输入 x_hsi: [B, 1, hsi_channels, P, P]
        self.conv3d_h1 = nn.Conv3d(1, 8, kernel_size=(7, 3, 3), padding=(3, 1, 1))
        self.conv3d_h2 = nn.Conv3d(8, 16, kernel_size=(5, 3, 3), padding=(2, 1, 1))
        self.conv3d_h3 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        # HSI 2D 卷积进一步处理
        self.conv2d_h1 = nn.Conv2d(32 * hsi_channels, 64, kernel_size=3, padding=1)
        self.conv2d_h2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # --- SAR 2D 卷积分支 ---
        # 输入 x_sar: [B, sar_channels, P, P]
        self.conv2d_s1 = nn.Conv2d(sar_channels, 32, kernel_size=3, padding=1)
        self.bn_s1 = nn.BatchNorm2d(32)
        self.conv2d_s2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn_s2 = nn.BatchNorm2d(32)

        # --- 特征融合 + 分类 ---
        # 融合后再 conv + 全局池化
        fused_channels = 64 + 32
        self.conv2d_f1 = nn.Conv2d(fused_channels, 64, kernel_size=3, padding=1)
        self.bn_f1 = nn.BatchNorm2d(64)
        self.adapt_pool = nn.AdaptiveAvgPool2d(1)

        # 全连接层
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x_hsi, x_sar):
        B = x_hsi.size(0)
        # --- HSI 分支 ---
        if x_hsi.dim() == 4:
            xh = x_hsi.unsqueeze(1)  # [B,1,C,P,P]
        else:
            xh = x_hsi
        xh = self.relu(self.conv3d_h1(xh))
        xh = self.relu(self.conv3d_h2(xh))
        xh = self.relu(self.conv3d_h3(xh))
        # xh: [B,32,C,P,P]
        B_, C3, L, H, W = xh.shape
        xh = xh.view(B_, C3 * L, H, W)  # [B, 32*L, P, P]
        xh = self.relu(self.conv2d_h1(xh))  # [B,64,P,P]
        xh = self.relu(self.conv2d_h2(xh))  # [B,64,P,P]

        # --- SAR 分支 ---
        xs = self.relu(self.bn_s1(self.conv2d_s1(x_sar)))  # [B,32,P,P]
        xs = self.relu(self.bn_s2(self.conv2d_s2(xs)))  # [B,32,P,P]

        # --- 融合 ---
        xf = torch.cat([xh, xs], dim=1)  # [B,96,P,P]
        xf = self.relu(self.bn_f1(self.conv2d_f1(xf)))  # [B,64,P,P]
        xf = self.adapt_pool(xf)  # [B,64,1,1]
        xf = xf.view(B, -1)  # [B,64]

        # --- 分类 ---
        xf = self.relu(self.fc1(xf))  # [B,256]
        logits = self.fc2(xf)  # [B,num_classes]
        return logits


if __name__ == "__main__":
    # 示例
    model = HybridSNMulti(hsi_channels=30, sar_channels=1, num_classes=10, patch_size=10)
    hsi_patch = torch.randn(128, 30, 10, 10)
    sar_patch = torch.randn(128, 1, 10, 10)
    out = model(hsi_patch, sar_patch)
    print("Output shape:", out.shape)  # [128,10]
