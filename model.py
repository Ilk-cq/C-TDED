import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0
import math
from typing import Dict, Tuple, List
import numpy as np


class HighResolutionFeatureExtractor(nn.Module):


    def __init__(self):
        super().__init__()

        backbone = efficientnet_b0(weights='DEFAULT')

        self.features = backbone.features
        original_conv = self.features[0][0]
        self.features[0][0] = nn.Conv2d(
            original_conv.in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=1,
            padding=original_conv.padding,
            bias=original_conv.bias
        )
        self.features[0][0].weight.data = original_conv.weight.data
        if original_conv.bias is not None:
            self.features[0][0].bias.data = original_conv.bias.data

        self._get_feature_channels()

        self.feature_layers = [0, 1, 2, 3, 4, 5, 6]

        self.high_res_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

    def _get_feature_channels(self):
        dummy_input = torch.randn(1, 3, 224, 224)
        self.feature_channels = {}
        x = dummy_input
        with torch.no_grad():
            for i, layer in enumerate(self.features):
                x = layer(x)
                self.feature_channels[i] = x.shape[1]

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


class DetailPreservingSideOutput(nn.Module):

    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.detail_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        self.edge_head = nn.Sequential(
            nn.Conv2d(in_channels // 4, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        detail_feat = self.detail_branch(x)
        attention = self.spatial_attention(detail_feat)
        attended = detail_feat * attention
        edge_out = self.edge_head(attended)
        return edge_out


class HighResolutionFPN(nn.Module):
    def __init__(self, feature_channels, output_dim=64):
        super().__init__()
        self.feature_channels = feature_channels
        self.output_dim = output_dim
        self.lateral_convs = nn.ModuleList()
        for channels in feature_channels:
            self.lateral_convs.append(
                nn.Conv2d(channels, output_dim, 1)
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
            nn.Conv2d(output_dim * len(feature_channels), output_dim, 3, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
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

        # 特征融合
        fused = torch.cat(fpn_features, dim=1)
        fused = self.fusion_conv(fused)

        return fused, fpn_features


class FineGrainedSemanticEdgeBranch(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = HighResolutionFeatureExtractor()

        self.feature_channels = [
            self.feature_extractor.feature_channels[i]
            for i in self.feature_extractor.feature_layers
        ]

        self.fpn = HighResolutionFPN(self.feature_channels, output_dim=64)

        self.side_outputs = nn.ModuleList([
            DetailPreservingSideOutput(channels)
            for channels in self.feature_channels
        ])

        self.high_res_fusion = nn.Sequential(
            nn.Conv2d(64 + 32, 64, 3, padding=1),  # FPN特征 + 原始高分辨率特征
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1)
        )

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
        edge_logits = self.final_conv(fused_features)

        return {
            'side_outputs': side_outputs,
            'edge_logits': edge_logits,
            'multi_scale_features': fused_features,
            'raw_features': features,
            'fpn_features': fpn_features
        }

class LightweightPatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=8, in_channels=3, embed_dim=64):  # 减小patch_size
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 4, 3, 1, 1),  # 步长从2改为1
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 4, embed_dim // 2, 3, 2, 1),  # 第一次下采样
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1),  # 第二次下采样
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 3, 2, 1)  # 第三次下采样
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


class GlobalContextBranch(nn.Module):
    def __init__(self, img_size=224, patch_size=8, in_channels=3, embed_dim=64, depth=4, num_heads=4):
        super().__init__()
        self.patch_embed = LightweightPatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
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

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.feature_mapper = nn.Sequential(
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )

    def forward(self, x, cnn_features=None):
        h, w = x.shape[2:]
        x, (patch_h, patch_w) = self.patch_embed(x)

        # 位置编码
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = torch.zeros(1, x.size(1), x.size(2), device=x.device)
            nn.init.trunc_normal_(pos_embed, std=0.02)
            x = x + pos_embed
        else:
            x = x + self.pos_embed

        x = self.dropout(x)

        # Transformer blocks
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


class MemoryEfficientFusion(nn.Module):
    def __init__(self, edge_dim=32, context_dim=32, fusion_dim=64):
        super().__init__()
        self.edge_proj = nn.Conv2d(edge_dim, fusion_dim, 1)
        self.context_proj = nn.Conv2d(context_dim, fusion_dim, 1)
        self.conv_attention = nn.Sequential(
            nn.Conv2d(fusion_dim * 2, fusion_dim // 2, 3, padding=1),
            nn.BatchNorm2d(fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_dim // 2, fusion_dim * 2, 1),
            nn.Sigmoid()
        )
        self.fusion_network = nn.Sequential(
            nn.Conv2d(fusion_dim * 2, fusion_dim, 3, padding=1),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_dim, fusion_dim // 2, 3, padding=1),
            nn.BatchNorm2d(fusion_dim // 2),
            nn.ReLU(inplace=True)
        )
        self.detail_enhance = nn.Sequential(
            nn.Conv2d(fusion_dim // 2, fusion_dim // 4, 3, padding=1),
            nn.BatchNorm2d(fusion_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_dim // 4, fusion_dim // 4, 3, padding=1),
            nn.BatchNorm2d(fusion_dim // 4),
            nn.ReLU(inplace=True)
        )
        self.edge_refine = nn.Sequential(
            nn.Conv2d(fusion_dim // 2 + fusion_dim // 4, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1)
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
    def count_parameters(model: nn.Module, detailed: bool = True) -> Dict[str, int]:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        result = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': total_params - trainable_params
        }

        if detailed:
            module_params = {}
            for name, module in model.named_children():
                module_params[name] = sum(p.numel() for p in module.parameters())
            result['module_params'] = module_params

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
    def calculate_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, float]:
        def conv_flops(layer, input_shape, output_shape):
            batch_size = input_shape[0]
            in_channels = layer.in_channels
            out_channels = layer.out_channels
            kernel_size = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
            out_h, out_w = output_shape[2], output_shape[3]
            multiplications = kernel_size * kernel_size * in_channels * out_channels * out_h * out_w * batch_size
            additions = (kernel_size * kernel_size * in_channels - 1) * out_channels * out_h * out_w * batch_size

            if layer.bias is not None:
                additions += out_channels * out_h * out_w * batch_size

            return multiplications + additions

        def linear_flops(layer, input_shape):
            batch_size = input_shape[0]
            in_features = layer.in_features
            out_features = layer.out_features

            multiplications = batch_size * in_features * out_features
            additions = batch_size * (in_features - 1) * out_features

            if layer.bias is not None:
                additions += batch_size * out_features

            return multiplications + additions

        def attention_flops(dim, num_heads, seq_len, batch_size):
            head_dim = dim // num_heads

            # Q, K, V projection
            qkv_flops = 3 * batch_size * seq_len * dim * dim

            # Attention scores
            scores_flops = batch_size * num_heads * seq_len * seq_len * head_dim

            # Softmax (approximation)
            softmax_flops = batch_size * num_heads * seq_len * seq_len * 5

            # Attention output
            attn_out_flops = batch_size * num_heads * seq_len * seq_len * head_dim

            # Output projection
            out_proj_flops = batch_size * seq_len * dim * dim

            return qkv_flops + scores_flops + softmax_flops + attn_out_flops + out_proj_flops

        total_flops = 0
        def register_hook(module):
            def hook(module, input, output):
                nonlocal total_flops

                if isinstance(module, nn.Conv2d):
                    if hasattr(input[0], 'shape'):
                        flops = conv_flops(module, input[0].shape, output.shape)
                        total_flops += flops

                elif isinstance(module, nn.Linear):
                    if hasattr(input[0], 'shape'):
                        flops = linear_flops(module, input[0].shape)
                        total_flops += flops

            if not isinstance(module, (nn.Sequential, nn.ModuleList)):
                module.register_forward_hook(hook)
        model.apply(register_hook)
        device = next(model.parameters()).device
        dummy_input = torch.randn(*input_shape).to(device)

        with torch.no_grad():
            model.eval()
            _ = model(dummy_input)
        for module in model.modules():
            module._forward_hooks.clear()
        gflops = total_flops / 1e9

        return {
            'total_flops': total_flops,
            'gflops': gflops,
            'mflops': total_flops / 1e6
        }

    @staticmethod
    def estimate_memory_usage(model: nn.Module, input_shape: Tuple[int, ...],
                              batch_size: int = 1, training: bool = True) -> Dict[str, float]:

        input_memory = np.prod(input_shape) * 4  # float32
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
    def profile_model(model: nn.Module, input_shape: Tuple[int, ...],
                      detailed: bool = True) -> Dict[str, any]:

        analyzer = ModelAnalyzer()
        results = {
            'input_shape': input_shape,
            'model_name': model.__class__.__name__,
        }
        param_stats = analyzer.count_parameters(model, detailed=detailed)
        results['parameters'] = param_stats
        model_size = analyzer.get_model_size(model, 'MB')
        results['model_size'] = model_size
        try:
            flops = analyzer.calculate_flops(model, input_shape)
            results['flops'] = flops
        except Exception as e:
            results['flops'] = {'error': str(e)}
        memory_train = analyzer.estimate_memory_usage(model, input_shape, training=True)
        memory_infer = analyzer.estimate_memory_usage(model, input_shape, training=False)
        results['memory'] = {
            'training': memory_train,
            'inference': memory_infer
        }

        return results

    @staticmethod
    def print_analysis(analysis_results: Dict[str, any]):

        print("\n" + "=" * 60)
        print(f"📊 Model Analysis Report: {analysis_results['model_name']}")
        print("=" * 60)


        print(f"\n📥 Input Shape: {analysis_results['input_shape']}")


        params = analysis_results['parameters']
        print(f"\n📈 Parameter Statistics:")
        print(f"   Total Parameters: {params['total_params']:,}")
        print(f"   Trainable Parameters: {params['trainable_params']:,}")
        print(f"   Non-trainable Parameters: {params['non_trainable_params']:,}")

        if 'module_params' in params:
            print(f"\n   Module-wise Parameters:")
            for name, count in params['module_params'].items():
                print(f"      {name}: {count:,}")


        size = analysis_results['model_size']
        print(f"\n💾 Model Size:")
        print(f"   Parameter Size: {size['param_size']:.2f} {size['unit']}")
        print(f"   Buffer Size: {size['buffer_size']:.2f} {size['unit']}")
        print(f"   Total Size: {size['total_size']:.2f} {size['unit']}")


        if 'error' not in analysis_results['flops']:
            flops = analysis_results['flops']
            print(f"\n⚡ Computational Complexity:")
            print(f"   Total FLOPs: {flops['total_flops']:,}")
            print(f"   GFLOPs: {flops['gflops']:.2f}")
            print(f"   MFLOPs: {flops['mflops']:.2f}")


        print(f"\n🧠 Memory Usage:")

        train_mem = analysis_results['memory']['training']
        print(f"\n   Training Mode:")
        print(f"      Input Memory: {train_mem['input_memory_mb']:.2f} MB")
        print(f"      Parameter Memory: {train_mem['param_memory_mb']:.2f} MB")
        print(f"      Gradient Memory: {train_mem['grad_memory_mb']:.2f} MB")
        print(f"      Activation Memory: {train_mem['activation_memory_mb']:.2f} MB")
        print(f"      Optimizer Memory: {train_mem['optimizer_memory_mb']:.2f} MB")
        print(f"      Total Memory: {train_mem['total_memory_gb']:.2f} GB")

        infer_mem = analysis_results['memory']['inference']
        print(f"\n   Inference Mode:")
        print(f"      Total Memory: {infer_mem['total_memory_gb']:.2f} GB")

        print("\n" + "=" * 60 + "\n")


class DualBranchEdgeModel(nn.Module):


    def __init__(self, img_size=224):
        super().__init__()
        self.fine_edge_branch = FineGrainedSemanticEdgeBranch()
        self.global_context_branch = GlobalContextBranch(img_size=img_size, patch_size=8)  # 减小patch_size
        self.fusion_module = MemoryEfficientFusion(edge_dim=32, context_dim=32)
        self.side_weights = nn.Parameter(torch.ones(7))  # 增加侧输出数量
        self.analyzer = ModelAnalyzer()

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

    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")

        edge_params = sum(p.numel() for p in self.fine_edge_branch.parameters())
        context_params = sum(p.numel() for p in self.global_context_branch.parameters())
        fusion_params = sum(p.numel() for p in self.fusion_module.parameters())

        print(f"边缘分支参数量(高分辨率EfficientNet-B2): {edge_params:,}")
        print(f"上下文分支参数量: {context_params:,}")
        print(f"融合模块参数量: {fusion_params:,}")
        return total_params

    def get_memory_usage(self, input_shape):
        total_memory = 0
        input_memory = 4 * input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]
        total_memory += input_memory
        param_memory = sum(p.numel() * 4 for p in self.parameters())
        total_memory += param_memory
        total_memory += param_memory
        activation_memory = input_memory * 15
        total_memory += activation_memory

        print(f"估算内存使用量(高分辨率版本): {total_memory / (1024 ** 3):.2f} GB")
        return total_memory

    def analyze_model(self, input_shape: Tuple[int, ...] = (1, 3, 224, 224),
                      print_results: bool = True) -> Dict[str, any]:
        analysis = self.analyzer.profile_model(self, input_shape, detailed=True)

        if print_results:
            self.analyzer.print_analysis(analysis)

        return analysis

    def get_model_size(self, format: str = 'MB') -> Dict[str, float]:
        return self.analyzer.get_model_size(self, format)

    def calculate_flops(self, input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> Dict[str, float]:
        return self.analyzer.calculate_flops(self, input_shape)

    def estimate_training_memory(self, batch_size: int = 4,
                                 input_size: int = 224) -> Dict[str, float]:
        input_shape = (batch_size, 3, input_size, input_size)
        return self.analyzer.estimate_memory_usage(self, input_shape, training=True)

    def estimate_inference_memory(self, batch_size: int = 1,
                                  input_size: int = 224) -> Dict[str, float]:
        input_shape = (batch_size, 3, input_size, input_size)
        return self.analyzer.estimate_memory_usage(self, input_shape, training=False)


def train_with_memory_optimization(model, dataloader, optimizer, criterion, device, scheduler=None):
    model.train()
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    total_loss = 0
    num_batches = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        if torch.rand(1) > 0.5:
            data = torch.flip(data, dims=[3])
            target = torch.flip(target, dims=[3])

        optimizer.zero_grad()

        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            outputs = model(data)
            loss, loss_dict = criterion(outputs, target)

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

        del outputs, loss_dict, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return total_loss / num_batches


def inference_with_sigmoid(model, x):
    model.eval()
    with torch.no_grad():
        outputs = model(x)

        # 应用sigmoid激活
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


