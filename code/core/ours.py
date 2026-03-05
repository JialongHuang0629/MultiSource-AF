import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Adaptive Upsample ----------
class AdaptiveUpsample(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor=2):
        super().__init__()
        self.conv    = nn.Conv2d(in_ch, out_ch * (scale_factor**2), 3, padding=1, bias=False)
        self.shuffle = nn.PixelShuffle(scale_factor)
        self.act     = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)
        return self.act(x)

# ---------- Transformer Fusion Module (如前) ----------
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.1):
        super().__init__()
        out_features    = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))
    
class LearnableHotSpot(nn.Module):
    def __init__(self, in_ch, temp=1.0):
        super().__init__()
        self.attn_conv = nn.Conv2d(in_ch, 1, 1)   # 1×1 注意力
        self.temp = temp
    def forward(self, x):
        # 1. 1×1 卷积 → 注意力图 [B,1,H,W]
        attn = self.attn_conv(x)
        # 2. 软最大化 → 总和为 1（热点权重）
        attn = F.softmax(attn.view(x.size(0), -1) / self.temp, dim=-1)
        # 3. 加权求和 → [B,C] 热点特征
        feat = (x.view(x.size(0), x.size(1), -1) * attn.unsqueeze(1)).sum(-1)
        return feat, attn.view(x.size(0), 1, x.size(2), x.size(3))

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, int(dim*mlp_ratio), dropout=dropout)
    def forward(self, x):
        res = x
        x, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + res
        res = x
        x   = self.mlp(self.norm2(x)) + res
        return x

class CrossModalityFusionWithTransformer(nn.Module):
    def __init__(self, in_ch_hsi, in_ch_sar, mid_ch=64, out_ch=64, trans_depth=1):
        super().__init__()
        self.hsi_conv = nn.Sequential(
            nn.Conv2d(in_ch_hsi, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True)
        )
        self.sar_conv = nn.Sequential(
            nn.Conv2d(in_ch_sar, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True)
        )
        self.attn_gen = nn.Sequential(
            nn.Conv2d(mid_ch*2, mid_ch, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, 1),
            nn.Sigmoid()
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(mid_ch*2, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
        self.transformers = nn.Sequential(*[
            TransformerBlock(dim=out_ch, num_heads=4) for _ in range(trans_depth)
        ])

    def forward(self, hsi, sar):
        fh = self.hsi_conv(hsi)  # [B, mid_ch, H, W]
        fs = self.sar_conv(sar)  # [B, mid_ch, H, W]

        cat = torch.cat([fh, fs], dim=1)       # [B, 2*mid_ch, H, W]
        attn_map = self.attn_gen(cat)          # [B, 1, H, W]
        fs_attn = fs * attn_map                # [B, mid_ch, H, W]

        fused = torch.cat([fh, fs_attn], dim=1)  # [B, 2*mid_ch, H, W]
        out   = self.fuse_conv(fused)            # [B, out_ch, H, W]

        B, C, H, W = out.shape
        x = out.flatten(2).transpose(1, 2)        # [B, H*W, C]
        x = self.transformers(x)                 # [B, H*W, C]
        out = x.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]
        return out, attn_map

class MultiSourceClassifier(nn.Module):
    def __init__(self, num_classes, scale=2, up_ch=64, mid_ch=64, out_ch=64, trans_depth=1):
        super().__init__()
        # 1) 自适应放大
        self.hsi_up = AdaptiveUpsample(30, up_ch, scale_factor=scale)
        self.sar_up = AdaptiveUpsample(4,  up_ch, scale_factor=scale)#Lidar需更改
        # 2) 跨模态融合 + Transformer 残差
        # self.fusion  = CrossModalityFusionWithTransformer(up_ch
        # )
        self.fusion  = CrossModalityFusionWithTransformer(
            in_ch_hsi=up_ch, in_ch_sar=up_ch,
            mid_ch=mid_ch, out_ch=out_ch,
            trans_depth=trans_depth
        )
        self.learnable_hotspot = LearnableHotSpot(64, temp=3.0)
        # 3) 中心—上下文 门控融合
        #    输入维度 = 2 * out_ch, 输出维度 = out_ch（每通道一个门控权重）
        self.gate_fc = nn.Linear(out_ch * 2, out_ch)
        # 4) 分类头
        self.classifier = nn.Sequential(
            nn.Linear(out_ch, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, hsi, sar):
        # Upsample
        uh = self.hsi_up(hsi)              # [B, up_ch, H', W']
        us = self.sar_up(sar)              # [B, up_ch, H', W']
        # Fusion + Transformer
        fused, attn_map = self.fusion(uh, us)  # [B, out_ch, H', W'], [B,1,H',W']

        B, C, H, W = fused.shape
        # --- 中心 + 上下文 特征提取 ---
        center_feat_fixed = fused[:, :, H//2, W//2] 
        center_feat, _  = self.learnable_hotspot(fused)
        center_conbine = (center_feat_fixed + center_feat) / 2.0               # [B, C]
        context_feat = F.adaptive_avg_pool2d(fused, 1).view(B, C)  # [B, C]

        # --- 门控融合 ---
        cat = torch.cat([center_conbine, context_feat], dim=1)  # [B, 2C]
        gate = torch.sigmoid(self.gate_fc(cat))             # [B, C], ∈(0,1)
        combined = gate * center_conbine + (1 - gate) * context_feat  # [B, C]

        # --- 分类 ---
        logits = self.classifier(combined)  # [B, num_classes]
        #return logits, attn_map
        return logits

# ---------------- Example ----------------
if __name__ == "__main__":
    model = MultiSourceClassifier(num_classes=10)
    hsi = torch.randn(4, 30, 10, 10)
    sar = torch.randn(4, 1, 10, 10)
    logits= model(hsi, sar)
    print("logits:", logits.shape)   # [4, 10]

