# -*- coding: utf-8 -*-
"""
可视化工具模块

提供缺陷检测结果可视化、训练过程绘图等功能。
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional
from pathlib import Path


def draw_defect_bounding_boxes(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    labels: Optional[List[str]] = None,
    scores: Optional[List[float]] = None,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """
    在图像上绘制缺陷区域边界框

    Args:
        image: 输入图像 (BGR格式)
        boxes: 边界框列表 [(x1, y1, x2, y2), ...]
        labels: 标签列表
        scores: 置信度列表
        color: 框颜色 (BGR)
        thickness: 框线粗细

    Returns:
        标注后的图像
    """
    result = image.copy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        # 添加标签
        if labels or scores:
            label = ""
            if labels and i < len(labels):
                label = labels[i]
            if scores and i < len(scores):
                label += f" {scores[i]:.2f}"

            if label:
                # 绘制标签背景
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    result,
                    (x1, y1 - label_h - 10),
                    (x1 + label_w, y1),
                    color,
                    -1,
                )
                # 绘制标签文字
                cv2.putText(
                    result,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

    return result


def highlight_defect_regions(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.5,
) -> np.ndarray:
    """
    高亮显示缺陷区域

    Args:
        image: 输入图像 (BGR格式)
        mask: 缺陷掩码 (二值图)
        color: 高亮颜色 (BGR)
        alpha: 透明度

    Returns:
        高亮后的图像
    """
    result = image.copy()

    # 创建彩色掩码
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color

    # 叠加
    result = cv2.addWeighted(result, 1, colored_mask, alpha, 0)

    return result


def visualize_prediction(
    image: np.ndarray,
    defect_score: float,
    threshold: float = 0.5,
    defect_mask: Optional[np.ndarray] = None,
    defect_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
) -> np.ndarray:
    """
    可视化预测结果

    Args:
        image: 输入图像 (BGR格式)
        defect_score: 缺陷概率
        threshold: 判定阈值
        defect_mask: 缺陷掩码
        defect_boxes: 缺陷边界框

    Returns:
        可视化图像
    """
    result = image.copy()
    h, w = image.shape[:2]

    # 添加标题背景
    title = f"Defect Score: {defect_score:.4f}"
    is_defect = defect_score > threshold
    status = "DEFECT" if is_defect else "NORMAL"

    # 状态颜色
    if is_defect:
        status_color = (0, 0, 255)  # 红色
    else:
        status_color = (0, 255, 0)  # 绿色

    # 绘制状态标签
    cv2.rectangle(result, (0, 0), (w, 50), status_color, -1)
    cv2.putText(
        result,
        f"{status} - {title}",
        (10, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # 如果有掩码，高亮显示
    if defect_mask is not None:
        result = highlight_defect_regions(result, defect_mask)

    # 如果有边界框，绘制
    if defect_boxes:
        result = draw_defect_bounding_boxes(result, defect_boxes)

    return result


def plot_training_history(
    history: dict,
    save_path: Optional[str] = None,
) -> None:
    """
    绘制训练历史曲线

    Args:
        history: 训练历史 {'loss': [...], 'val_loss': [...], 'accuracy': [...]}
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    if "loss" in history:
        axes[0].plot(history["loss"], label="Train Loss")
        if "val_loss" in history:
            axes[0].plot(history["val_loss"], label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        axes[0].legend()
        axes[0].grid(True)

    # 准确率曲线
    if "accuracy" in history:
        axes[1].plot(history["accuracy"], label="Train Accuracy")
        if "val_accuracy" in history:
            axes[1].plot(history["val_accuracy"], label="Val Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training and Validation Accuracy")
        axes[1].legend()
        axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_confusion_matrix(
    confusion_mat: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
) -> None:
    """
    绘制混淆矩阵

    Args:
        confusion_mat: 混淆矩阵
        class_names: 类别名称
        save_path: 保存路径
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        confusion_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def create_result_grid(
    images: List[np.ndarray],
    titles: Optional[List[str]] = None,
    rows: int = 2,
    cols: int = 3,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    创建图像网格

    Args:
        images: 图像列表
        titles: 标题列表
        rows: 行数
        cols: 列数
        save_path: 保存路径

    Returns:
        网格图像
    """
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten()

    for idx, img in enumerate(images):
        if idx >= len(axes):
            break

        # BGR转RGB用于显示
        if img.ndim == 3 and img.shape[2] == 3:
            img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_display = img

        axes[idx].imshow(img_display)
        if titles and idx < len(titles):
            axes[idx].set_title(titles[idx])
        axes[idx].axis("off")

    # 隐藏多余的子图
    for idx in range(len(images), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.close()
