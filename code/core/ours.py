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

# ---------- Transformer Fusion Module  ----------
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

class PseudoWaveletConv(nn.Module):
    """
    可直接替换旧版的小波卷积模块
    参数:
        channels (int): 输入输出通道数
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.kernel_size = 3
        self.padding = self.kernel_size // 2
        self.level = 2  # 与旧模块一致，只做一级分解

        # 注册 Haar 小波滤波器（不参与训练）
        self.register_buffer('ll', torch.tensor([[[[1, 1], [1, 1]]]], dtype=torch.float32))
        self.register_buffer('lh', torch.tensor([[[[1, 1], [-1, -1]]]], dtype=torch.float32))
        self.register_buffer('hl', torch.tensor([[[[1, -1], [1, -1]]]], dtype=torch.float32))
        self.register_buffer('hh', torch.tensor([[[[1, -1], [-1, 1]]]], dtype=torch.float32))

        # 为每一级创建四个 depthwise 卷积（这里只有一级）
        self.level_convs = nn.ModuleList()
        for _ in range(self.level):
            convs = nn.ModuleDict({
                'll': nn.Conv2d(channels, channels, self.kernel_size,
                                padding=self.padding, groups=channels, bias=False),
                'lh': nn.Conv2d(channels, channels, self.kernel_size,
                                padding=self.padding, groups=channels, bias=False),
                'hl': nn.Conv2d(channels, channels, self.kernel_size,
                                padding=self.padding, groups=channels, bias=False),
                'hh': nn.Conv2d(channels, channels, self.kernel_size,
                                padding=self.padding, groups=channels, bias=False)
            })
            self.level_convs.append(convs)

        # 1x1 卷积用于调整输出通道数（此处输入输出相同，但保留以匹配结构）
        self.proj = nn.Conv2d(channels, channels, 1)

        # 残差连接（通道相同，直接恒等映射）
        self.shortcut = None  # 因为 in_channels == out_channels

    def _haar_decomp(self, x):
        """单级 Haar 分解，返回四个子带 (B, C, H/2, W/2)"""
        B, C, H, W = x.shape
        ll = F.conv2d(x, self.ll.repeat(C, 1, 1, 1), stride=2, groups=C)
        lh = F.conv2d(x, self.lh.repeat(C, 1, 1, 1), stride=2, groups=C)
        hl = F.conv2d(x, self.hl.repeat(C, 1, 1, 1), stride=2, groups=C)
        hh = F.conv2d(x, self.hh.repeat(C, 1, 1, 1), stride=2, groups=C)
        return ll, lh, hl, hh

    def _haar_recomp(self, ll, lh, hl, hh):
        """单级 Haar 重构，返回 (B, C, H*2, W*2)"""
        B, C, H_half, W_half = ll.shape
        ll_rec = F.conv_transpose2d(ll, self.ll.repeat(C, 1, 1, 1), stride=2, groups=C)
        lh_rec = F.conv_transpose2d(lh, self.lh.repeat(C, 1, 1, 1), stride=2, groups=C)
        hl_rec = F.conv_transpose2d(hl, self.hl.repeat(C, 1, 1, 1), stride=2, groups=C)
        hh_rec = F.conv_transpose2d(hh, self.hh.repeat(C, 1, 1, 1), stride=2, groups=C)
        return (ll_rec + lh_rec + hl_rec + hh_rec) / 4.0

    def forward(self, x):
        orig_h, orig_w = x.shape[2], x.shape[3]

        # 确保尺寸能被 2^level 整除（这里 level=1，只需偶数）
        pad_h = (2 - orig_h % 2) % 2
        pad_w = (2 - orig_w % 2) % 2
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # 小波分解
        ll, lh, hl, hh = self._haar_decomp(x)

        # 对每个子带应用可学习卷积（depthwise）
        conv = self.level_convs[0]  # 只有一级
        ll_conv = conv['ll'](ll)
        lh_conv = conv['lh'](lh)
        hl_conv = conv['hl'](hl)
        hh_conv = conv['hh'](hh)

        # 重构
        recon = self._haar_recomp(ll_conv, lh_conv, hl_conv, hh_conv)

        # 投影到输出通道
        out = self.proj(recon)

        # 残差连接
        out = out + x

        # 如果之前有填充，裁剪回原始尺寸
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :orig_h, :orig_w]

        return out


class CrossModalityFusionWithWavelet(nn.Module):
    """
    在原有架构中插入小波变换模块
    """
    def __init__(self, in_ch_hsi, in_ch_sar, mid_ch=64, out_ch=64, trans_depth=1):
        super().__init__()
        
        # 原有卷积保持不变
        self.hsi_conv = nn.Sequential(
            nn.Conv2d(in_ch_hsi, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True)
        )
        self.sar_conv = nn.Sequential(
            nn.Conv2d(in_ch_sar, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True)
        )
        
        # 新增：小波增强模块
        self.hsi_wavelet = PseudoWaveletConv(mid_ch)
        self.sar_wavelet = PseudoWaveletConv(mid_ch)
        
        # 小波域注意力生成（比原空间域更鲁棒）
        self.wavelet_attn = nn.Sequential(
            nn.Conv2d(mid_ch * 2, mid_ch, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, 1),
            nn.Sigmoid()
        )
        
        # 后续融合和Transformer保持原样
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(mid_ch * 2, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
        
        self.transformers = nn.Sequential(*[
            TransformerBlock(dim=out_ch, num_heads=4) for _ in range(trans_depth)
        ])

    def forward(self, hsi, sar):
        # 基础特征提取
        fh = self.hsi_conv(hsi)  
        fs = self.sar_conv(sar)  
        
        # 小波增强（分离频率信息）
        fh_wt = self.hsi_wavelet(fh)  # 增强后的HSI特征
        fs_wt = self.sar_wavelet(fs)  # 增强后的SAR特征
        
        # 基于小波特征的注意力（对噪声更鲁棒）
        cat_wt = torch.cat([fh_wt, fs_wt], dim=1)
        attn_map = self.wavelet_attn(cat_wt)
        
        # SAR高频信息加权（利用SAR的结构优势）
        fs_attn = fs_wt * attn_map
        
        # 融合
        fused = torch.cat([fh_wt, fs_attn], dim=1)
        out = self.fuse_conv(fused)

        B, C, H, W = out.shape
        x = out.flatten(2).transpose(1, 2)
        
        # Transformer
        x = self.transformers(x) 
        
        out = x.transpose(1, 2).view(B, C, H, W)
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
        self.fusion  = CrossModalityFusionWithWavelet(
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

