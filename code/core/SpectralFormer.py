import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralFormerMulti(nn.Module):
    def __init__(self,
                 hsi_channels: int,
                 sar_channels: int,
                 patch_size: int,
                 num_classes: int,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 6,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1):
        """
        Multi-source SpectralFormer: HSI + SAR patch classification.
        """
        super(SpectralFormerMulti, self).__init__()
        self.patch_size = patch_size
        # HSI Transformer settings
        self.hsi_channels = hsi_channels
        self.proj = nn.Linear(patch_size * patch_size, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, hsi_channels + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        # SAR convolutional branch
        self.conv_s1 = nn.Conv2d(sar_channels, 32, kernel_size=3, padding=1, bias=False)
        self.bn_s1 = nn.BatchNorm2d(32)
        self.conv_s2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn_s2 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Fusion classifier
        fused_dim = d_model + 64
        self.fc1 = nn.Linear(fused_dim, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, num_classes)

    def forward(self, x_hsi: torch.Tensor, x_sar: torch.Tensor) -> torch.Tensor:
        # x_hsi: [B, C_h, P, P]
        B, C_h, P, _ = x_hsi.shape
        # Flatten spatial dims: [B, C_h, P*P]
        x = x_hsi.view(B, C_h, -1)
        # Project each band-patch to d_model: [B, C_h, d_model]
        x = self.proj(x)
        # Prepend class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, x], dim=1)  # [B, C_h+1, d_model]
        tokens = tokens + self.pos_embed
        # Transformer encoding
        encoded = self.transformer(tokens)  # [B, C_h+1, d_model]
        hsi_feat = self.dropout(encoded[:, 0])  # class token feature

        # SAR branch
        s = self.relu(self.bn_s1(self.conv_s1(x_sar)))  # [B,32,P,P]
        s = self.relu(self.bn_s2(self.conv_s2(s)))      # [B,64,P,P]
        s = self.pool(s).view(B, -1)                    # [B,64]

        # Fusion and classification
        fused = torch.cat([hsi_feat, s], dim=1)         # [B, d_model+64]
        out = self.relu(self.fc1(fused))                # [B, dim_feedforward]
        logits = self.fc2(out)                          # [B, num_classes]
        return logits

if __name__ == "__main__":
    model = SpectralFormerMulti(hsi_channels=30, sar_channels=1, patch_size=10, num_classes=7)
    hsi_patch = torch.randn(8, 30, 10, 10)
    sar_patch = torch.randn(8, 1, 10, 10)
    out = model(hsi_patch, sar_patch)
    print("Output shape:", out.shape)  # [8,10]

