# -*- coding: utf-8 -*-
"""
数据增强模块

提供多种图像增强方法，包括翻转、裁剪、亮度调整等。
"""

import cv2
import numpy as np
import random
from typing import Tuple, Optional


class DataAugmentor:
    """数据增强器类"""

    def __init__(self, seed: Optional[int] = None):
        """
        初始化增强器

        Args:
            seed: 随机种子，用于 reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def horizontal_flip(self, image: np.ndarray, prob: float = 0.5) -> np.ndarray:
        """
        水平翻转

        Args:
            image: 输入图像
            prob: 翻转概率

        Returns:
            翻转后的图像
        """
        if random.random() < prob:
            return cv2.flip(image, 1)
        return image

    def vertical_flip(self, image: np.ndarray, prob: float = 0.5) -> np.ndarray:
        """
        垂直翻转

        Args:
            image: 输入图像
            prob: 翻转概率

        Returns:
            翻转后的图像
        """
        if random.random() < prob:
            return cv2.flip(image, 0)
        return image

    def random_rotation(
        self,
        image: np.ndarray,
        angle_range: Tuple[float, float] = (-15, 15),
    ) -> np.ndarray:
        """
        随机旋转

        Args:
            image: 输入图像
            angle_range: 旋转角度范围 (min, max)

        Returns:
            旋转后的图像
        """
        angle = random.uniform(*angle_range)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # 获取旋转矩阵
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 执行旋转
        return cv2.warpAffine(
            image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT
        )

    def random_crop(
        self,
        image: np.ndarray,
        crop_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        随机裁剪

        Args:
            image: 输入图像
            crop_size: 裁剪尺寸 (height, width)

        Returns:
            裁剪后的图像
        """
        h, w = image.shape[:2]
        crop_h, crop_w = crop_size

        if h <= crop_h or w <= crop_w:
            # 如果图像小于裁剪尺寸，使用resize
            return cv2.resize(image, (crop_w, crop_h))

        # 随机选择裁剪位置
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        # 裁剪
        cropped = image[top : top + crop_h, left : left + crop_w]

        return cv2.resize(cropped, (w, h))

    def random_brightness(
        self,
        image: np.ndarray,
        factor_range: Tuple[float, float] = (0.7, 1.3),
    ) -> np.ndarray:
        """
        随机亮度调整

        Args:
            image: 输入图像
            factor_range: 亮度调整因子范围

        Returns:
            调整后的图像
        """
        factor = random.uniform(*factor_range)
        adjusted = np.clip(image * factor, 0, 255).astype(np.uint8)
        return adjusted

    def random_contrast(
        self,
        image: np.ndarray,
        factor_range: Tuple[float, float] = (0.7, 1.3),
    ) -> np.ndarray:
        """
        随机对比度调整

        Args:
            image: 输入图像
            factor_range: 对比度调整因子范围

        Returns:
            调整后的图像
        """
        factor = random.uniform(*factor_range)
        mean = image.mean()
        adjusted = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
        return adjusted

    def random_noise(
        self,
        image: np.ndarray,
        noise_level: float = 10,
    ) -> np.ndarray:
        """
        添加随机噪声

        Args:
            image: 输入图像
            noise_level: 噪声强度

        Returns:
            添加噪声后的图像
        """
        noise = np.random.randn(*image.shape) * noise_level
        noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
        return noisy

    def random_blur(
        self,
        image: np.ndarray,
        kernel_size_range: Tuple[int, int] = (3, 7),
        prob: float = 0.3,
    ) -> np.ndarray:
        """
        随机模糊

        Args:
            image: 输入图像
            kernel_size_range: 卷积核大小范围
            prob: 应用模糊的概率

        Returns:
            模糊后的图像
        """
        if random.random() < prob:
            kernel_size = random.choice(
                range(kernel_size_range[0], kernel_size_range[1] + 1, 2)
            )
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return image

    def random_erode_dilate(
        self,
        image: np.ndarray,
        prob: float = 0.3,
    ) -> np.ndarray:
        """
        随机形态学操作（腐蚀/膨胀）

        Args:
            image: 输入图像
            prob: 操作概率

        Returns:
            处理后的图像
        """
        if random.random() < prob:
            kernel = np.ones((3, 3), np.uint8)
            if random.random() < 0.5:
                return cv2.erode(image, kernel, iterations=1)
            else:
                return cv2.dilate(image, kernel, iterations=1)
        return image

    def augment(
        self,
        image: np.ndarray,
        enable_all: bool = True,
        flip_prob: float = 0.5,
        brightness_range: Tuple[float, float] = (0.7, 1.3),
        rotation_range: Tuple[float, float] = (-15, 15),
        blur_prob: float = 0.3,
    ) -> np.ndarray:
        """
        综合数据增强（训练时使用）

        Args:
            image: 输入图像
            enable_all: 是否启用所有增强
            flip_prob: 翻转概率
            brightness_range: 亮度范围
            rotation_range: 旋转角度范围
            blur_prob: 模糊概率

        Returns:
            增强后的图像
        """
        # 翻转
        image = self.horizontal_flip(image, prob=flip_prob)
        image = self.vertical_flip(image, prob=flip_prob * 0.5)

        # 旋转
        image = self.random_rotation(image, angle_range=rotation_range)

        # 亮度调整
        image = self.random_brightness(image, factor_range=brightness_range)

        # 对比度调整
        image = self.random_contrast(image, factor_range=brightness_range)

        # 模糊
        image = self.random_blur(image, prob=blur_prob)

        return image


def create_train_augmentor(seed: int = 42) -> DataAugmentor:
    """
    创建训练数据增强器

    Args:
        seed: 随机种子

    Returns:
        DataAugmentor实例
    """
    return DataAugmentor(seed=seed)


def create_val_augmentor() -> DataAugmentor:
    """
    创建验证/测试数据增强器（仅做基础处理）

    Returns:
        DataAugmentor实例
    """
    return DataAugmentor(seed=None)
