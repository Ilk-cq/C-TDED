import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0
import math
from typing import Dict, Tuple, List, Optional, Union
import numpy as np
try:
    from thop import profile, clever_format

    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

try:
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False


class EdgePreservingModule(nn.Module):


    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels


        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 3, padding=4, dilation=4),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1),  # 1x1 卷积保持原始信息
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )


        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


        self.adapt = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):

        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        feat4 = self.branch4(x)


        multi_scale = torch.cat([feat1, feat2, feat3, feat4], dim=1)


        fused = self.fusion(multi_scale)


        return fused + self.adapt(x)


class AdaptiveHighResolutionFeatureExtractor(nn.Module):


    def __init__(self):
        super().__init__()

        backbone = efficientnet_b0(weights='DEFAULT')
        self.features = backbone.features

        original_conv = self.features[0][0]
        self.features[0][0] = nn.Conv2d(
            original_conv.in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=1,  # 从2改为1，不进行初始下采样
            padding=original_conv.padding,
            bias=original_conv.bias
        )
        self.features[0][0].weight.data = original_conv.weight.data
        if original_conv.bias is not None:
            self.features[0][0].bias.data = original_conv.bias.data


        self._get_feature_channels()

        self.feature_layers = [0, 1, 2, 3, 4, 5, 6, 7]  # 使用更多层

        self.high_res_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            EdgePreservingModule(32, 32),  # 添加边缘保持模块
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

    def _get_feature_channels(self):

        test_sizes = [(224, 224), (320, 480), (480, 320)]
        self.feature_channels = {}

        for h, w in test_sizes:
            dummy_input = torch.randn(1, 3, h, w)
            x = dummy_input
            with torch.no_grad():
                for i, layer in enumerate(self.features):
                    x = layer(x)
                    if i not in self.feature_channels:
                        self.feature_channels[i] = x.shape[1]
            break

    def forward(self, x):
        h, w = x.shape[2:]
        features = []
        current_x = x

        high_res_feature = self.high_res_conv(x)

        for i, layer in enumerate(self.features):
            current_x = layer(current_x)

            if i in self.feature_layers:
                features.append(current_x)

        return features, high_res_feature


class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ImprovedDetailPreservingSideOutput(nn.Module):

    def __init__(self, in_channels, out_channels=1):
        super().__init__()

        self.edge_enhance = EdgePreservingModule(in_channels, in_channels // 2)

        self.detail_branch = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 4, 1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid()
        )

        self.edge_refine = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels // 8, 3, padding=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):

        edge_enhanced = self.edge_enhance(x)

        detail_feat = self.detail_branch(edge_enhanced)

        attention = self.spatial_attention(detail_feat)
        attended = detail_feat * attention

        edge_out = self.edge_refine(attended)

        return edge_out


class AdaptiveHighResolutionFPN(nn.Module):


    def __init__(self, feature_channels, output_dim=64):
        super().__init__()
        self.feature_channels = feature_channels
        self.output_dim = output_dim

        self.lateral_convs = nn.ModuleList()
        for channels in feature_channels:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(channels, output_dim, 1),
                    nn.BatchNorm2d(output_dim),
                    nn.ReLU(inplace=True),
                    EdgePreservingModule(output_dim, output_dim)  # 添加边缘保持
                )
            )

        self.output_convs = nn.ModuleList()
        for _ in feature_channels:
            self.output_convs.append(
                nn.Sequential(
                    nn.Conv2d(output_dim, output_dim, 3, padding=1),
                    nn.BatchNorm2d(output_dim),
                    nn.ReLU(inplace=True)
                )
            )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(output_dim * len(feature_channels), output_dim * 2, 3, padding=1),
            nn.BatchNorm2d(output_dim * 2),
            nn.ReLU(inplace=True),
            EdgePreservingModule(output_dim * 2, output_dim),  # 边缘保持
            nn.Conv2d(output_dim, output_dim, 3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, features, target_size):

        fpn_features = []

        for i, (feature, lateral_conv, output_conv) in enumerate(
                zip(features, self.lateral_convs, self.output_convs)
        ):
            lateral_feat = lateral_conv(feature)

            if lateral_feat.shape[-2:] != target_size:
                lateral_feat = F.interpolate(
                    lateral_feat, size=target_size,
                    mode='bilinear', align_corners=False
                )

            fpn_feat = output_conv(lateral_feat)
            fpn_features.append(fpn_feat)

        fused = torch.cat(fpn_features, dim=1)
        fused = self.fusion_conv(fused)

        return fused, fpn_features


class EnhancedEdgeRefinementModule(nn.Module):

    def __init__(self, in_channels, mid_channels=32, out_channels=1):
        super().__init__()

        self.coarse_edge = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            EdgePreservingModule(mid_channels, mid_channels)
        )

        self.fine_edge = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels // 2, 1),
            nn.BatchNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, mid_channels // 2, 3, padding=1),
            nn.BatchNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True)
        )

        fusion_in_channels = mid_channels + mid_channels // 2

        self.edge_fusion = nn.Sequential(
            nn.Conv2d(fusion_in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels // 2, 3, padding=1),
            nn.BatchNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, out_channels, 1)
        )

    def forward(self, x):
        coarse = self.coarse_edge(x)
        fine = self.fine_edge(x)

        combined = torch.cat([coarse, fine], dim=1)
        refined = self.edge_fusion(combined)

        return refined


class AdaptiveFineGrainedSemanticEdgeBranch(nn.Module):


    def __init__(self):
        super().__init__()

        self.feature_extractor = AdaptiveHighResolutionFeatureExtractor()

        self.feature_channels = [
            self.feature_extractor.feature_channels[i]
            for i in self.feature_extractor.feature_layers
        ]

        self.fpn = AdaptiveHighResolutionFPN(self.feature_channels, output_dim=64)

        self.side_outputs = nn.ModuleList([
            ImprovedDetailPreservingSideOutput(channels)
            for channels in self.feature_channels
        ])

        self.high_res_fusion = nn.Sequential(
            nn.Conv2d(64 + 32, 96, 3, padding=1),  # FPN特征 + 原始高分辨率特征
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            EdgePreservingModule(96, 64),  # 添加边缘保持
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 【新增】边缘细化模块
        self.edge_refinement = EnhancedEdgeRefinementModule(32, 32, 1)

    def forward(self, x):
        h, w = x.shape[2:]

        features, high_res_feature = self.feature_extractor(x)

        fpn_fused, fpn_features = self.fpn(features, (h, w))

        side_outputs = []
        for i, (feature, side_out_module) in enumerate(zip(features, self.side_outputs)):
            side_out = side_out_module(feature)

            if side_out.shape[-2:] != (h, w):
                side_out = F.interpolate(
                    side_out, size=(h, w),
                    mode='bilinear', align_corners=False
                )
            side_outputs.append(side_out)

        combined_features = torch.cat([fpn_fused, high_res_feature], dim=1)
        fused_features = self.high_res_fusion(combined_features)

        edge_logits = self.edge_refinement(fused_features)

        return {
            'side_outputs': side_outputs,
            'edge_logits': edge_logits,
            'multi_scale_features': fused_features,
            'raw_features': features,
            'fpn_features': fpn_features
        }


class AdaptiveLightweightPatchEmbedding(nn.Module):

    def __init__(self, patch_size=8, in_channels=3, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 4, 3, 1, 1),  # 步长1，不下采样
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 4, embed_dim // 2, 3, 2, 1),  # 第一次下采样 (1/2)
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            EdgePreservingModule(embed_dim // 2, embed_dim // 2),  # 边缘保持
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1),  # 第二次下采样 (1/4)
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 3, 2, 1)  # 第三次下采样 (1/8)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.projection(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (H, W)


class EfficientLocalAttention(nn.Module):

    def __init__(self, dim, window_size=7, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def window_partition(self, x, window_size):
        B, H, W, C = x.shape
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            H_padded, W_padded = H + pad_h, W + pad_w
        else:
            H_padded, W_padded = H, W
        x = x.view(B, H_padded // window_size, window_size, W_padded // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows, H, W, H_padded, W_padded

    def window_reverse(self, windows, window_size, H, W, H_padded, W_padded):
        B = int(windows.shape[0] / (H_padded * W_padded / window_size / window_size))
        x = windows.view(B, H_padded // window_size, W_padded // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_padded, W_padded, -1)
        if H_padded != H or W_padded != W:
            x = x[:, :H, :W, :]
        return x

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        if H * W != N:
            H = int(math.sqrt(N))
            W = N // H
            while H * W != N and H > 1:
                H -= 1
                W = N // H

        x = x.reshape(B, H, W, C)
        x_windows, H_orig, W_orig, H_padded, W_padded = self.window_partition(x, self.window_size)
        x_windows = x_windows.reshape(-1, self.window_size * self.window_size, C)
        qkv = self.qkv(x_windows).reshape(-1, self.window_size * self.window_size, 3, self.num_heads,
                                          self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        x_windows = (attn @ v).transpose(1, 2).reshape(-1, self.window_size * self.window_size, C)
        x_windows = self.proj(x_windows)
        x_windows = self.dropout(x_windows)
        x_windows = x_windows.reshape(-1, self.window_size, self.window_size, C)
        x = self.window_reverse(x_windows, self.window_size, H_orig, W_orig, H_padded, W_padded)
        x = x.reshape(B, N, C)
        return x


class LightweightTransformerBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=2.0, dropout=0.1, window_size=7):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientLocalAttention(dim, window_size, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class EnhancedCrossAttentionModule(nn.Module):

    def __init__(self, query_dim=64, key_value_dim=32, num_heads=4, dropout=0.1):
        super().__init__()

        self.multi_scale_fusion = nn.Sequential(
            nn.Linear(query_dim + query_dim, query_dim),
            nn.LayerNorm(query_dim),
            nn.GELU(),
            nn.Linear(query_dim, query_dim)
        )

        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_value_dim, query_dim)
        self.v_proj = nn.Linear(key_value_dim, query_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(query_dim, query_dim)
        self.proj_dropout = nn.Dropout(dropout)

        self.local_enhance = nn.Sequential(
            nn.Linear(query_dim, query_dim // 2),
            nn.GELU(),
            nn.Linear(query_dim // 2, query_dim),
            nn.LayerNorm(query_dim)
        )

        self.detail_branch = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.LayerNorm(query_dim),
            nn.ReLU(inplace=True)
        )

        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

    def forward(self, query, key_value):
        if key_value.size(-1) != query.size(-1):
            key_value = self.k_proj(key_value)

        combined = torch.cat([query, key_value], dim=-1)
        fused = self.multi_scale_fusion(combined)

        local_enhanced = self.local_enhance(fused)

        detail_enhanced = self.detail_branch(local_enhanced)

        output = fused + local_enhanced + detail_enhanced

        return output


class AdaptiveGlobalContextBranch(nn.Module):

    def __init__(self, patch_size=8, in_channels=3, embed_dim=64, depth=4, num_heads=4):
        super().__init__()
        self.patch_embed = AdaptiveLightweightPatchEmbedding(patch_size, in_channels, embed_dim)
        self.dropout = nn.Dropout(0.1)

        self.blocks = nn.ModuleList([
            LightweightTransformerBlock(embed_dim, num_heads, mlp_ratio=2.0, dropout=0.1, window_size=7)
            for _ in range(depth)
        ])

        self.cross_attention = EnhancedCrossAttentionModule(
            query_dim=embed_dim,
            key_value_dim=32,
            num_heads=4,
            dropout=0.1
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.context_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 32)
        )

        self.feature_mapper = nn.Sequential(
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )

        self.register_buffer('pos_embed_cache', torch.zeros(1, 1, embed_dim))

    def get_positional_embedding(self, N, device):
        if N != self.pos_embed_cache.size(1):
            pos_embed = torch.zeros(1, N, self.pos_embed_cache.size(2), device=device)
            nn.init.trunc_normal_(pos_embed, std=0.02)
            return pos_embed
        return self.pos_embed_cache[:, :N]

    def forward(self, x, cnn_features=None):
        h, w = x.shape[2:]
        x, (patch_h, patch_w) = self.patch_embed(x)

        pos_embed = self.get_positional_embedding(x.size(1), x.device)
        x = x + pos_embed
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        if cnn_features is not None and isinstance(cnn_features, dict) and 'multi_scale_features' in cnn_features:
            cnn_feat = cnn_features['multi_scale_features']
            B, C, H, W = cnn_feat.shape
            cnn_feat = cnn_feat.flatten(2).transpose(1, 2)
            cnn_feat = self.feature_mapper(cnn_feat)

            if cnn_feat.size(1) != x.size(1):
                B_cnn, N_cnn, C_cnn = cnn_feat.shape
                H_cnn = W_cnn = int(math.sqrt(N_cnn))

                if H_cnn * W_cnn != N_cnn:
                    H_cnn = int(math.sqrt(N_cnn))
                    W_cnn = N_cnn // H_cnn
                    while H_cnn * W_cnn != N_cnn and H_cnn > 1:
                        H_cnn -= 1
                        W_cnn = N_cnn // H_cnn

                cnn_feat_spatial = cnn_feat.transpose(1, 2).reshape(B_cnn, C_cnn, H_cnn, W_cnn)

                cnn_feat_spatial = F.interpolate(
                    cnn_feat_spatial,
                    size=(patch_h, patch_w),
                    mode='bilinear',
                    align_corners=False
                )
                cnn_feat = cnn_feat_spatial.flatten(2).transpose(1, 2)

            x = self.cross_attention(x, cnn_feat)

        x = self.norm(x)
        global_context = self.context_head(x)
        context_feature_map = global_context.transpose(1, 2).reshape(x.shape[0], -1, patch_h, patch_w)

        return {
            'global_context_features': context_feature_map,
            'patch_features': global_context
        }


class AdaptiveMemoryEfficientFusion(nn.Module):

    def __init__(self, edge_dim=32, context_dim=32, fusion_dim=64):
        super().__init__()
        self.edge_proj = nn.Conv2d(edge_dim, fusion_dim, 1)
        self.context_proj = nn.Conv2d(context_dim, fusion_dim, 1)

        # 【优化】增强的注意力机制
        self.conv_attention = nn.Sequential(
            nn.Conv2d(fusion_dim * 2, fusion_dim, 3, padding=1),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True),
            EdgePreservingModule(fusion_dim, fusion_dim // 2),  # 边缘保持
            nn.Conv2d(fusion_dim // 2, fusion_dim * 2, 1),
            nn.Sigmoid()
        )

        self.fusion_network = nn.Sequential(
            nn.Conv2d(fusion_dim * 2, fusion_dim, 3, padding=1),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True),
            EdgePreservingModule(fusion_dim, fusion_dim),  # 边缘保持
            nn.Conv2d(fusion_dim, fusion_dim // 2, 3, padding=1),
            nn.BatchNorm2d(fusion_dim // 2),
            nn.ReLU(inplace=True)
        )

        self.detail_enhance = nn.Sequential(
            nn.Conv2d(fusion_dim // 2, fusion_dim // 4, 3, padding=1),
            nn.BatchNorm2d(fusion_dim // 4),
            nn.ReLU(inplace=True),
            EdgePreservingModule(fusion_dim // 4, fusion_dim // 4),
            nn.Conv2d(fusion_dim // 4, fusion_dim // 4, 3, padding=1),
            nn.BatchNorm2d(fusion_dim // 4),
            nn.ReLU(inplace=True)
        )

        final_in_channels = fusion_dim // 2 + fusion_dim // 4
        self.edge_refine = nn.Sequential(
            nn.Conv2d(final_in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, edge_features, context_features):
        h, w = edge_features.shape[2:]
        edge_proj = self.edge_proj(edge_features)
        context_proj = F.interpolate(
            self.context_proj(context_features),
            size=(h, w), mode='bilinear', align_corners=False
        )

        concat_features = torch.cat([edge_proj, context_proj], dim=1)
        attention_weight = self.conv_attention(concat_features)
        attended_features = concat_features * attention_weight

        fused_features = self.fusion_network(attended_features)

        detail_features = self.detail_enhance(fused_features)

        final_features = torch.cat([fused_features, detail_features], dim=1)
        final_output = self.edge_refine(final_features)

        return final_output


class ModelAnalyzer:
    @staticmethod
    def format_params(num: int) -> str:

        return f"{num / 1e6:.2f} M"

    @staticmethod
    def format_number(num: float, precision: int = 2) -> str:
        if num >= 1e12:
            return f"{num / 1e12:.{precision}f} T"
        elif num >= 1e9:
            return f"{num / 1e9:.{precision}f} G"
        elif num >= 1e6:
            return f"{num / 1e6:.{precision}f} M"
        elif num >= 1e3:
            return f"{num / 1e3:.{precision}f} K"
        else:
            return f"{num:.{precision}f}"

    @staticmethod
    def count_parameters(model: nn.Module, detailed: bool = True) -> Dict[str, any]:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        result = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': non_trainable_params,
            'total_params_str': ModelAnalyzer.format_params(total_params),
            'trainable_params_str': ModelAnalyzer.format_params(trainable_params),
            'non_trainable_params_str': ModelAnalyzer.format_params(non_trainable_params),
        }

        if detailed:
            module_params = {}
            module_params_str = {}
            for name, module in model.named_children():
                params = sum(p.numel() for p in module.parameters())
                module_params[name] = params
                module_params_str[name] = ModelAnalyzer.format_params(params)
            result['module_params'] = module_params
            result['module_params_str'] = module_params_str

        return result

    @staticmethod
    def get_model_size(model: nn.Module, format: str = 'MB') -> Dict[str, float]:
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.numel() * 4

        for buffer in model.buffers():
            buffer_size += buffer.numel() * 4

        total_size = param_size + buffer_size

        divisor = {
            'B': 1,
            'KB': 1024,
            'MB': 1024 ** 2,
            'GB': 1024 ** 3
        }.get(format, 1024 ** 2)

        return {
            'param_size': param_size / divisor,
            'buffer_size': buffer_size / divisor,
            'total_size': total_size / divisor,
            'unit': format
        }

    @staticmethod
    def calculate_flops_macs(model: nn.Module, input_shape: Tuple[int, ...],
                             device: str = 'cpu') -> Dict[str, any]:

        model_copy = model.to(device)
        model_copy.eval()

        dummy_input = torch.randn(*input_shape).to(device)

        result = {
            'flops': None,
            'macs': None,
            'flops_str': 'N/A',
            'macs_str': 'N/A',
            'method': None,
            'detailed_flops': None
        }

        if THOP_AVAILABLE:
            try:
                with torch.no_grad():
                    macs, params = profile(model_copy, inputs=(dummy_input,), verbose=False)
                flops = macs * 2  # MACs to FLOPs

                result['macs'] = int(macs)
                result['flops'] = int(flops)
                result['macs_str'] = ModelAnalyzer.format_number(macs)
                result['flops_str'] = ModelAnalyzer.format_number(flops)
                result['method'] = 'thop'

                return result
            except Exception as e:
                print(f"thop计算失败: {e}")

        if FVCORE_AVAILABLE:
            try:
                with torch.no_grad():
                    flops_analysis = FlopCountAnalysis(model_copy, dummy_input)
                    flops = flops_analysis.total()
                    macs = flops / 2

                result['flops'] = int(flops)
                result['macs'] = int(macs)
                result['flops_str'] = ModelAnalyzer.format_number(flops)
                result['macs_str'] = ModelAnalyzer.format_number(macs)
                result['method'] = 'fvcore'
                result['detailed_flops'] = flop_count_table(flops_analysis)

                return result
            except Exception as e:
                print(f"fvcore计算失败: {e}")

        try:
            flops, macs = ModelAnalyzer._manual_flops_count(model_copy, input_shape, device)
            result['flops'] = int(flops)
            result['macs'] = int(macs)
            result['flops_str'] = ModelAnalyzer.format_number(flops)
            result['macs_str'] = ModelAnalyzer.format_number(macs)
            result['method'] = 'manual'
        except Exception as e:
            print(f"手动计算失败: {e}")
            result['method'] = 'failed'

        return result

    @staticmethod
    def _manual_flops_count(model: nn.Module, input_shape: Tuple[int, ...],
                            device: str = 'cpu') -> Tuple[int, int]:
        total_flops = 0
        total_macs = 0

        def conv_flops_hook(module, input, output):
            nonlocal total_flops, total_macs

            batch_size = input[0].size(0)
            output_dims = output.shape[2:]

            kernel_dims = module.kernel_size
            in_channels = module.in_channels
            out_channels = module.out_channels
            groups = module.groups

            filters_per_channel = out_channels // groups
            conv_per_position_macs = int(np.prod(kernel_dims)) * (in_channels // groups) * filters_per_channel

            active_elements_count = batch_size * int(np.prod(output_dims))

            macs = conv_per_position_macs * active_elements_count
            flops = 2 * macs

            if module.bias is not None:
                flops += out_channels * active_elements_count

            total_macs += macs
            total_flops += flops

        def bn_flops_hook(module, input, output):
            nonlocal total_flops
            total_flops += input[0].numel() * 2

        def relu_flops_hook(module, input, output):
            nonlocal total_flops
            total_flops += input[0].numel()

        def linear_flops_hook(module, input, output):
            nonlocal total_flops, total_macs

            batch_size = input[0].size(0) if input[0].dim() > 1 else 1

            macs = module.in_features * module.out_features * batch_size
            flops = 2 * macs

            if module.bias is not None:
                flops += module.out_features * batch_size

            total_macs += macs
            total_flops += flops

        def pooling_flops_hook(module, input, output):
            nonlocal total_flops
            total_flops += output.numel()

        hooks = []

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(conv_flops_hook))
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                hooks.append(module.register_forward_hook(bn_flops_hook))
            elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.GELU, nn.Sigmoid)):
                hooks.append(module.register_forward_hook(relu_flops_hook))
            elif isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(linear_flops_hook))
            elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                hooks.append(module.register_forward_hook(pooling_flops_hook))

        model = model.to(device)
        model.eval()
        dummy_input = torch.randn(*input_shape).to(device)

        with torch.no_grad():
            model(dummy_input)

        for hook in hooks:
            hook.remove()

        return total_flops, total_macs

    @staticmethod
    def estimate_memory_usage(model: nn.Module, input_shape: Tuple[int, ...],
                              batch_size: int = 1, training: bool = True) -> Dict[str, float]:
        input_memory = np.prod(input_shape) * 4
        param_memory = sum(p.numel() * 4 for p in model.parameters())
        grad_memory = param_memory if training else 0
        activation_memory = param_memory * 15 if training else param_memory * 5
        optimizer_memory = param_memory * 2 if training else 0
        total_memory = input_memory + param_memory + grad_memory + activation_memory + optimizer_memory

        return {
            'input_memory_mb': input_memory / (1024 ** 2),
            'param_memory_mb': param_memory / (1024 ** 2),
            'grad_memory_mb': grad_memory / (1024 ** 2),
            'activation_memory_mb': activation_memory / (1024 ** 2),
            'optimizer_memory_mb': optimizer_memory / (1024 ** 2),
            'total_memory_mb': total_memory / (1024 ** 2),
            'total_memory_gb': total_memory / (1024 ** 3)
        }

    @staticmethod
    def full_analysis(model: nn.Module, input_shape: Tuple[int, ...],
                      device: str = 'cpu', print_results: bool = True) -> Dict[str, any]:
        analysis = {
            'input_shape': input_shape,
            'parameters': ModelAnalyzer.count_parameters(model, detailed=True),
            'model_size': ModelAnalyzer.get_model_size(model, 'MB'),
            'flops_macs': ModelAnalyzer.calculate_flops_macs(model, input_shape, device),
            'memory': {
                'training': ModelAnalyzer.estimate_memory_usage(model, input_shape, training=True),
                'inference': ModelAnalyzer.estimate_memory_usage(model, input_shape, training=False)
            }
        }

        if print_results:
            ModelAnalyzer._print_full_analysis(analysis)

        return analysis

    @staticmethod
    def _print_full_analysis(analysis: Dict[str, any]):
        print("\n" + "=" * 70)
        print("📊 Complete Model Analysis")
        print("=" * 70)
        print(f"Input Shape: {analysis['input_shape']}")

        params = analysis['parameters']
        print(f"\n📈 Parameters:")
        print(f"  Total:         {params['total_params_str']:>12}")
        print(f"  Trainable:     {params['trainable_params_str']:>12}")
        print(f"  Non-trainable: {params['non_trainable_params_str']:>12}")

        if 'module_params_str' in params:
            print(f"\n  Module breakdown:")
            for name, count_str in params['module_params_str'].items():
                print(f"    {name}: {count_str}")

        size = analysis['model_size']
        print(f"\n💾 Model Size: {size['total_size']:.2f} {size['unit']}")

        flops_macs = analysis['flops_macs']
        print(f"\n⚡ Computational Complexity (method: {flops_macs['method']}):")
        print(f"  FLOPs: {flops_macs['flops_str']}")
        print(f"  MACs:  {flops_macs['macs_str']}")

        mem = analysis['memory']
        print(f"\n🧠 Memory Usage:")
        print(f"  Training:  {mem['training']['total_memory_gb']:.2f} GB")
        print(f"  Inference: {mem['inference']['total_memory_gb']:.2f} GB")
        print("=" * 70)


class AdaptiveDualBranchEdgeModel(nn.Module):


    def __init__(self, img_size: Optional[Union[int, Tuple[int, int]]] = None):

        super().__init__()

        self.fine_edge_branch = AdaptiveFineGrainedSemanticEdgeBranch()
        self.global_context_branch = AdaptiveGlobalContextBranch(patch_size=8)
        self.fusion_module = AdaptiveMemoryEfficientFusion(edge_dim=32, context_dim=32)

        self.side_weights = nn.Parameter(torch.ones(len(self.fine_edge_branch.side_outputs)))

        self.analyzer = ModelAnalyzer()

        self.img_size = img_size
        self.adaptive_mode = (img_size is None)

    def forward(self, x):


        edge_results = self.fine_edge_branch(x)
        context_results = self.global_context_branch(x, edge_results)
        final_edges = self.fusion_module(
            edge_results['multi_scale_features'],
            context_results['global_context_features']
        )

        return {
            'final_edges': final_edges,
            'side_outputs': edge_results['side_outputs'],
            'edge_logits': edge_results['edge_logits'],
            'global_context': context_results['global_context_features'],
            'side_weights': torch.softmax(self.side_weights, dim=0)
        }

    def process_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        images = batch['image']


        if 'target_size' in batch:

            outputs = self.forward(images)
            return outputs
        else:

            return self.forward(images)

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"总参数量: {total_params / 1e6:.2f} M")
        print(f"可训练参数量: {trainable_params / 1e6:.2f} M")

        edge_params = sum(p.numel() for p in self.fine_edge_branch.parameters())
        context_params = sum(p.numel() for p in self.global_context_branch.parameters())
        fusion_params = sum(p.numel() for p in self.fusion_module.parameters())

        print(f"边缘分支参数量: {edge_params / 1e6:.2f} M")
        print(f"上下文分支参数量: {context_params / 1e6:.2f} M")
        print(f"融合模块参数量: {fusion_params / 1e6:.2f} M")
        return total_params

    def analyze_model(self, input_shape: Optional[Tuple[int, ...]] = None,
                      print_results: bool = True,
                      device: str = 'cpu') -> Dict[str, any]:

        if input_shape is None:
            if self.img_size is not None:
                if isinstance(self.img_size, int):
                    input_shape = (1, 3, self.img_size, self.img_size)
                else:
                    input_shape = (1, 3, self.img_size[0], self.img_size[1])
            else:
                input_shape = (1, 3, 320, 480)

        analysis = {
            'input_shape': input_shape,
            'model_name': self.__class__.__name__,
            'adaptive_mode': self.adaptive_mode,
            'parameters': self.analyzer.count_parameters(self, detailed=True),
            'model_size': self.analyzer.get_model_size(self, 'MB'),
            'flops_macs': self.analyzer.calculate_flops_macs(self, input_shape, device),
            'memory': {
                'training': self.analyzer.estimate_memory_usage(self, input_shape, training=True),
                'inference': self.analyzer.estimate_memory_usage(self, input_shape, training=False)
            }
        }

        if print_results:
            self._print_analysis(analysis)

        return analysis

    def _print_analysis(self, analysis: Dict[str, any]):

        print("\n" + "=" * 70)
        print(f"📊 Model Analysis: {analysis['model_name']}")
        print("=" * 70)
        print(f"Mode: {'Adaptive' if analysis['adaptive_mode'] else 'Fixed'} Size")
        print(f"Input Shape: {analysis['input_shape']}")

        params = analysis['parameters']
        print(f"\n📈 Parameters:")
        print(f"  Total:         {params['total_params_str']:>12}")
        print(f"  Trainable:     {params['trainable_params_str']:>12}")
        print(f"  Non-trainable: {params['non_trainable_params_str']:>12}")

        if 'module_params_str' in params:
            print(f"\n  Module breakdown:")
            for name, count_str in params['module_params_str'].items():
                print(f"    {name}: {count_str}")

        size = analysis['model_size']
        print(f"\n💾 Model Size: {size['total_size']:.2f} {size['unit']}")

        flops_macs = analysis['flops_macs']
        print(f"\n⚡ Computational Complexity (method: {flops_macs['method']}):")
        print(f"  FLOPs: {flops_macs['flops_str']}")
        print(f"  MACs:  {flops_macs['macs_str']}")

        mem = analysis['memory']
        print(f"\n🧠 Memory Usage:")
        print(f"  Training:  {mem['training']['total_memory_gb']:.2f} GB")
        print(f"  Inference: {mem['inference']['total_memory_gb']:.2f} GB")
        print("=" * 70)


DualBranchEdgeModel = AdaptiveDualBranchEdgeModel


def train_with_adaptive_sizes(model, dataloader, optimizer, criterion, device, scheduler=None):

    model.train()
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    total_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        targets = batch['edge_map'].to(device)

        optimizer.zero_grad()

        # 混合精度训练
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            outputs = model.process_batch({'image': images})
            loss, loss_dict = criterion(outputs, targets)

        # 反向传播
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        # 内存清理
        if torch.cuda.is_available() and batch_idx % 10 == 0:
            torch.cuda.empty_cache()

    return total_loss / num_batches


def inference_with_sigmoid(model, x):

    model.eval()
    with torch.no_grad():

        if isinstance(x, dict):
            outputs = model.process_batch(x)
        else:
            outputs = model(x)

        final_prob = torch.sigmoid(outputs['final_edges'])
        side_probs = [torch.sigmoid(side_out) for side_out in outputs['side_outputs']]
        edge_prob = torch.sigmoid(outputs['edge_logits'])

        return {
            'final_edges': final_prob,
            'side_outputs': side_probs,
            'edge_probability': edge_prob,
            'global_context': outputs['global_context'],
            'side_weights': outputs['side_weights']
        }

