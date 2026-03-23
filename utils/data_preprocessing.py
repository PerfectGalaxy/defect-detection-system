# -*- coding: utf-8 -*-
"""
图像预处理工具模块

提供图像读取、归一化、尺寸调整等基础预处理功能。
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from pathlib import Path


class ImagePreprocessor:
    """图像预处理器类"""

    # ImageNet统计量（用于归一化）
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        mean: Optional[list] = None,
        std: Optional[list] = None,
    ):
        """
        初始化预处理器

        Args:
            target_size: 目标图像尺寸 (height, width)
            normalize: 是否进行归一化
            mean: 自定义均值
            std: 自定义标准差
        """
        self.target_size = target_size
        self.normalize = normalize
        self.mean = np.array(mean or self.IMAGENET_MEAN)
        self.std = np.array(std or self.IMAGENET_STD)

    def load_image(self, image_path: str) -> np.ndarray:
        """
        加载图像文件

        Args:
            image_path: 图像文件路径

        Returns:
            BGR格式的图像数组
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")
        return image

    def resize(self, image: np.ndarray) -> np.ndarray:
        """
        调整图像尺寸

        Args:
            image: 输入图像 (BGR格式)

        Returns:
            调整后的图像
        """
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)

    def bgr_to_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        BGR转RGB通道顺序

        Args:
            image: BGR格式图像

        Returns:
            RGB格式图像
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        归一化图像到[0, 1]，然后标准化

        Args:
            image: RGB格式图像，像素值范围[0, 255]

        Returns:
            归一化后的图像
        """
        # 转换为float32并归一化到[0, 1]
        image = image.astype(np.float32) / 255.0

        if self.normalize:
            # 标准化
            image = (image - self.mean) / self.std

        return image

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        完整预处理流程

        Args:
            image: 输入图像 (BGR格式)

        Returns:
            预处理后的图像 tensor ready for model
        """
        # 调整尺寸
        image = self.resize(image)

        # BGR转RGB
        image = self.bgr_to_rgb(image)

        # 归一化
        image = self.normalize_image(image)

        # 转换为CHW格式 (HWC -> CHW)
        image = np.transpose(image, (2, 0, 1))

        return image

    def preprocess_from_path(self, image_path: str) -> np.ndarray:
        """
        从文件路径预处理图像

        Args:
            image_path: 图像文件路径

        Returns:
            预处理后的图像
        """
        image = self.load_image(image_path)
        return self.preprocess(image)

    def denormalize(self, image: np.ndarray) -> np.ndarray:
        """
        反归一化（用于可视化）

        Args:
            image: 归一化后的图像

        Returns:
            可视化的图像 [0, 255]
        """
        if self.normalize:
            image = image * self.std + self.mean
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        return image


def load_image_as_tensor(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    便捷函数：从文件加载并预处理图像

    Args:
        image_path: 图像文件路径
        target_size: 目标尺寸

    Returns:
        预处理后的numpy数组
    """
    preprocessor = ImagePreprocessor(target_size=target_size)
    return preprocessor.preprocess_from_path(image_path)
