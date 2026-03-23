# -*- coding: utf-8 -*-
"""
合成测试数据生成脚本

生成模拟的工业零件表面图像，包含正常和缺陷样本，用于模型训练和测试。
"""

import os
import random
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFilter


class SyntheticDefectGenerator:
    """合成缺陷图像生成器"""

    # 缺陷类型
    DEFECT_TYPES = ["scratch", "crack", "stain", "dent"]

    def __init__(
        self,
        image_size: int = 224,
        output_dir: str = "data",
    ):
        """
        初始化生成器

        Args:
            image_size: 输出图像尺寸
            output_dir: 输出目录
        """
        self.image_size = image_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _generate_base_image(self) -> np.ndarray:
        """
        生成基础金属表面图像

        Returns:
            基础图像 (BGR)
        """
        # 创建金属质感背景
        image = np.random.randint(180, 220, (self.image_size, self.image_size, 3), dtype=np.uint8)

        # 添加噪点模拟金属纹理
        noise = np.random.normal(0, 15, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 添加一些纹理线条
        for _ in range(random.randint(5, 15)):
            x1 = random.randint(0, self.image_size)
            y1 = random.randint(0, self.image_size)
            x2 = random.randint(0, self.image_size)
            y2 = random.randint(0, self.image_size)
            color = random.randint(160, 240)
            cv2.line(image, (x1, y1), (x2, y2), (color, color, color), 1)

        # 平滑处理
        image = cv2.GaussianBlur(image, (3, 3), 0)

        return image

    def _add_scratch(self, image: np.ndarray) -> np.ndarray:
        """
        添加划痕缺陷

        Args:
            image: 基础图像

        Returns:
            添加划痕后的图像
        """
        # 随机起点和终点
        num_scratches = random.randint(1, 3)
        for _ in range(num_scratches):
            x1 = random.randint(0, self.image_size)
            y1 = random.randint(0, self.image_size)
            length = random.randint(30, 100)
            angle = random.uniform(0, 360)

            # 计算终点
            x2 = int(x1 + length * np.cos(np.radians(angle)))
            y2 = int(y1 + length * np.sin(np.radians(angle)))

            # 限制在图像范围内
            x2 = max(0, min(self.image_size - 1, x2))
            y2 = max(0, min(self.image_size - 1, y2))

            # 绘制划痕（深色线条）
            thickness = random.randint(1, 3)
            color = random.randint(50, 100)
            cv2.line(image, (x1, y1), (x2, y2), (color, color, color), thickness)

        return image

    def _add_crack(self, image: np.ndarray) -> np.ndarray:
        """
        添加裂纹缺陷

        Args:
            image: 基础图像

        Returns:
            添加裂纹后的图像
        """
        # 创建折线裂纹
        points = []
        x, y = random.randint(20, self.image_size - 20), random.randint(20, self.image_size - 20)
        num_segments = random.randint(3, 6)

        for _ in range(num_segments):
            points.append((x, y))
            x += random.randint(-30, 30)
            y += random.randint(-30, 30)
            x = max(0, min(self.image_size - 1, x))
            y = max(0, min(self.image_size - 1, y))

        # 绘制裂纹
        for i in range(len(points) - 1):
            cv2.line(
                image,
                points[i],
                points[i + 1],
                (random.randint(30, 60), random.randint(30, 60), random.randint(30, 60)),
                random.randint(1, 2),
            )

        return image

    def _add_stain(self, image: np.ndarray) -> np.ndarray:
        """
        添加污渍缺陷

        Args:
            image: 基础图像

        Returns:
            添加污渍后的图像
        """
        # 随机位置和大小
        x = random.randint(30, self.image_size - 30)
        y = random.randint(30, self.image_size - 30)
        radius = random.randint(15, 40)

        # 创建模糊的污渍区域
        mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        cv2.circle(mask, (x, y), radius, 255, -1)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)

        # 应用污渍颜色
        stain_color = random.randint(80, 140)
        for c in range(3):
            image[:, :, c] = np.where(
                mask > 50,
                np.minimum(image[:, :, c].astype(np.uint16), stain_color).astype(np.uint8),
                image[:, :, c],
            )

        return image

    def _add_dent(self, image: np.ndarray) -> np.ndarray:
        """
        添加凹坑/变形缺陷

        Args:
            image: 基础图像

        Returns:
            添加凹坑后的图像
        """
        x = random.randint(30, self.image_size - 30)
        y = random.randint(30, self.image_size - 30)
        radius = random.randint(20, 35)

        # 创建渐变的凹坑效果
        for r in range(radius, 0, -1):
            intensity = int(150 * (1 - r / radius))
            cv2.circle(image, (x, y), r, (intensity, intensity, intensity), -1)

        # 边缘加深
        cv2.circle(image, (x, y), radius, (80, 80, 80), 2)

        return image

    def _add_defect(self, image: np.ndarray, defect_type: str) -> np.ndarray:
        """
        添加指定类型的缺陷

        Args:
            image: 基础图像
            defect_type: 缺陷类型

        Returns:
            添加缺陷后的图像
        """
        if defect_type == "scratch":
            return self._add_scratch(image)
        elif defect_type == "crack":
            return self._add_crack(image)
        elif defect_type == "stain":
            return self._add_stain(image)
        elif defect_type == "dent":
            return self._add_dent(image)
        else:
            return image

    def generate_normal_image(self) -> np.ndarray:
        """
        生成正常图像

        Returns:
            正常图像
        """
        return self._generate_base_image()

    def generate_defect_image(self) -> np.ndarray:
        """
        生成缺陷图像

        Returns:
            缺陷图像
        """
        # 先生成基础图像
        image = self._generate_base_image()

        # 随机选择缺陷类型组合
        defect_types = random.sample(self.DEFECT_TYPES, k=random.randint(1, 2))

        for defect_type in defect_types:
            image = self._add_defect(image, defect_type)

        return image

    def generate_dataset(
        self,
        num_normal: int = 10,
        num_defect: int = 10,
        split: str = "val",
    ):
        """
        生成数据集

        Args:
            num_normal: 正常图像数量
            num_defect: 缺陷图像数量
            split: 数据集划分 ('train', 'val', 'test')
        """
        # 创建目录
        normal_dir = self.output_dir / split / "normal"
        defect_dir = self.output_dir / split / "defect"

        normal_dir.mkdir(parents=True, exist_ok=True)
        defect_dir.mkdir(parents=True, exist_ok=True)

        print(f"生成 {split} 数据集...")

        # 生成正常图像
        print(f"生成 {num_normal} 张正常图像...")
        for i in tqdm(range(num_normal)):
            image = self.generate_normal_image()
            # 添加一些变化
            if random.random() < 0.3:
                image = cv2.GaussianBlur(image, (5, 5), 0)
            if random.random() < 0.3:
                image = cv2.convertScaleAbs(image, alpha=random.uniform(0.9, 1.1), beta=0)

            cv2.imwrite(str(normal_dir / f"normal_{i:04d}.jpg"), image)

        # 生成缺陷图像
        print(f"生成 {num_defect} 张缺陷图像...")
        for i in tqdm(range(num_defect)):
            image = self.generate_defect_image()
            # 添加一些变化
            if random.random() < 0.3:
                image = cv2.GaussianBlur(image, (3, 3), 0)

            cv2.imwrite(str(defect_dir / f"defect_{i:04d}.jpg"), image)

        print(f"数据集生成完成！")
        print(f"  正常图像: {normal_dir}")
        print(f"  缺陷图像: {defect_dir}")


def generate_single_test_image(output_path: str, is_defect: bool = False):
    """
    生成单张测试图像

    Args:
        output_path: 输出路径
        is_defect: 是否为缺陷图像
    """
    generator = SyntheticDefectGenerator()

    if is_defect:
        image = generator.generate_defect_image()
    else:
        image = generator.generate_normal_image()

    cv2.imwrite(output_path, image)
    print(f"已生成测试图像: {output_path}")


def main():
    """主函数"""
    # 创建生成器
    generator = SyntheticDefectGenerator(image_size=224, output_dir="data")

    # 生成训练集
    generator.generate_dataset(num_normal=20, num_defect=20, split="train")

    # 生成验证集
    generator.generate_dataset(num_normal=5, num_defect=5, split="val")

    # 生成测试集
    generator.generate_dataset(num_normal=5, num_defect=5, split="test")

    print("\n所有数据集生成完成！")


if __name__ == "__main__":
    main()
