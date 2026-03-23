# -*- coding: utf-8 -*-
"""
缺陷检测模型模块

提供基于CNN和ViT的缺陷检测模型实现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class DefectConvNet(nn.Module):
    """
    轻量级CNN缺陷检测模型

    基于改进的MobileNetV2架构，适合工业场景的实时推理
    """

    def __init__(
        self,
        num_classes: int = 2,
        input_size: Tuple[int, int] = (224, 224),
        dropout: float = 0.3,
    ):
        """
        初始化模型

        Args:
            num_classes: 分类类别数
            input_size: 输入图像尺寸
            dropout: Dropout比例
        """
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        # 特征提取 backbone
        self.features = nn.Sequential(
            # Conv1: 224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            # Conv2: 112x112 -> 56x56
            self._make_expand_layer(32, 16, 1, stride=1),
            # Conv3: 56x56
            self._make_expand_layer(16, 24, 2, stride=2),
            # Conv4: 28x28
            self._make_expand_layer(24, 32, 3, stride=2),
            # Conv5: 14x14
            self._make_expand_layer(32, 64, 4, stride=2),
            # Conv6: 7x7
            self._make_expand_layer(64, 96, 3, stride=2),
            # Conv7: 3x3
            self._make_expand_layer(96, 160, 3, stride=2),
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(160, 64, kernel_size=1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(64, 160, kernel_size=1),
            nn.Sigmoid(),
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(160, 128),
            nn.ReLU6(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

        # 初始化权重
        self._initialize_weights()

    def _make_expand_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """创建扩展层"""
        layers = []
        for i in range(num_layers):
            layer_stride = stride if i == 0 else 1
            expand_ratio = 6

            # 逐层连接
            if expand_ratio > 1:
                layers.append(
                    self._expand_block(
                        in_channels,
                        out_channels,
                        expand_ratio,
                        layer_stride,
                    )
                )
            else:
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            3,
                            layer_stride,
                            1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU6(inplace=True),
                    )
                )

            in_channels = out_channels

        return nn.Sequential(*layers)

    def _expand_block(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int,
        stride: int,
    ) -> nn.Sequential:
        """扩展块"""
        hidden_dim = in_channels * expand_ratio

        layers = []
        if expand_ratio != 1:
            # 逐点卷积 - 扩展
            layers.append(
                nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False)
            )
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        # 深度卷积
        layers.append(
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
        )
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))

        # 逐点卷积 - 投影
        layers.append(
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False)
        )
        layers.append(nn.BatchNorm2d(out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入张量 (B, 3, H, W)

        Returns:
            (logits, attention_weights)
        """
        # 特征提取
        x = self.features(x)

        # 注意力机制
        attention_weights = self.attention(x)
        x = x * attention_weights

        # 全局池化
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)

        # 分类
        logits = self.classifier(x)

        return logits, attention_weights


class SimpleViTDefectDetector(nn.Module):
    """
    简化的Vision Transformer缺陷检测模型

    使用Patch Embedding + Transformer Encoder结构
    """

    def __init__(
        self,
        num_classes: int = 2,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        """
        初始化ViT模型

        Args:
            num_classes: 分类数
            image_size: 输入图像尺寸
            patch_size: Patch大小
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            num_layers: Transformer层数
            mlp_ratio: MLP扩展比例
            dropout: Dropout比例
        """
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Patch Embedding
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # 分类token和位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 分类头
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

        # 初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module)

    def _init_module(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入 (B, 3, H, W)

        Returns:
            分类logits (B, num_classes)
        """
        B = x.size(0)

        # Patch Embedding: (B, 3, H, W) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # 添加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer编码
        x = self.transformer(x)

        # 使用CLS token的输出进行分类
        cls_output = x[:, 0]
        logits = self.head(cls_output)

        return logits


def create_model(
    model_type: str = "cnn",
    num_classes: int = 2,
    pretrained: bool = False,
    **kwargs,
) -> nn.Module:
    """
    模型创建工厂函数

    Args:
        model_type: 模型类型 ('cnn' 或 'vit')
        num_classes: 分类数
        pretrained: 是否加载预训练权重
        **kwargs: 其他模型参数

    Returns:
        模型实例
    """
    if model_type.lower() == "cnn":
        model = DefectConvNet(
            num_classes=num_classes,
            input_size=kwargs.get("input_size", (224, 224)),
            dropout=kwargs.get("dropout", 0.3),
        )
    elif model_type.lower() == "vit":
        model = SimpleViTDefectDetector(
            num_classes=num_classes,
            image_size=kwargs.get("image_size", 224),
            patch_size=kwargs.get("patch_size", 16),
            embed_dim=kwargs.get("embed_dim", 256),
            num_heads=kwargs.get("num_heads", 8),
            num_layers=kwargs.get("num_layers", 6),
            dropout=kwargs.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    return model


class DefectDetector:
    """
    缺陷检测器封装类

    提供便捷的推理接口
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        threshold: float = 0.5,
    ):
        """
        初始化检测器

        Args:
            model: PyTorch模型
            device: 推理设备
            threshold: 判定阈值
        """
        self.model = model
        self.device = device
        self.threshold = threshold
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        image: torch.Tensor,
        return_probs: bool = False,
    ) -> Dict:
        """
        预测

        Args:
            image: 输入图像张量 (1, 3, H, W)
            return_probs: 是否返回概率

        Returns:
            预测结果字典
        """
        image = image.to(self.device)

        # 前向传播
        if isinstance(self.model, DefectConvNet):
            logits, attention = self.model(image)
        else:
            logits = self.model(image)
            attention = None

        probs = F.softmax(logits, dim=1)
        defect_prob = probs[0, 1].item()  # 缺陷类的概率

        result = {
            "is_defect": defect_prob > self.threshold,
            "defect_probability": defect_prob,
            "class": 1 if defect_prob > self.threshold else 0,
        }

        if return_probs:
            result["probabilities"] = probs.cpu().numpy().tolist()

        if attention is not None:
            result["attention"] = attention.cpu().numpy()

        return result


def count_parameters(model: nn.Module) -> int:
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: nn.Module) -> float:
    """获取模型大小 (MB)"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024
