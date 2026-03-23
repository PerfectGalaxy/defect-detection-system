# -*- coding: utf-8 -*-
"""
模型训练脚本

提供完整的训练流程，包括数据加载、模型训练、验证和模型保存。
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
import yaml

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models import create_model, count_parameters, get_model_size
from utils import create_train_augmentor, create_val_augmentor


class DefectDataset(Dataset):
    """缺陷检测数据集"""

    CLASSES = ["normal", "defect"]

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform=None,
        augmentor=None,
    ):
        """
        初始化数据集

        Args:
            data_dir: 数据目录
            split: 数据集划分 ('train', 'val', 'test')
            transform: 基础变换
            augmentor: 数据增强器
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.augmentor = augmentor

        # 加载数据
        self.image_paths = []
        self.labels = []

        # 扫描目录结构: data_dir/split/class_name/*.jpg
        split_dir = self.data_dir / split
        if split_dir.exists():
            for class_idx, class_name in enumerate(self.CLASSES):
                class_dir = split_dir / class_name
                if class_dir.exists():
                    for img_path in class_dir.glob("*.jpg"):
                        self.image_paths.append(str(img_path))
                        self.labels.append(class_idx)
                    for img_path in class_dir.glob("*.png"):
                        self.image_paths.append(str(img_path))
                        self.labels.append(class_idx)

        print(f"[{split}] 数据集加载完成: {len(self.image_paths)} 张图像")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = np.array(image)

        # 数据增强
        if self.split == "train" and self.augmentor:
            image = self.augmentor.augment(image)

        # 基础变换
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label


def get_transforms(image_size: int = 224, split: str = "train"):
    """获取数据变换"""
    if split == "train":
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch,
    print_freq: int = 10,
):
    """训练一个epoch"""
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 打印进度
        if (batch_idx + 1) % print_freq == 0:
            avg_loss = running_loss / (batch_idx + 1)
            acc = 100.0 * correct / total
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{acc:.2f}%"})

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(dataloader)
    val_acc = 100.0 * correct / total

    return val_loss, val_acc, all_preds, all_labels


def save_checkpoint(
    model,
    optimizer,
    epoch,
    best_acc,
    save_path,
    is_best: bool = False,
):
    """保存检查点"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_acc": best_acc,
    }
    torch.save(checkpoint, save_path)

    if is_best:
        best_path = str(Path(save_path).parent / "best_model.pth")
        torch.save(checkpoint, best_path)
        print(f"已保存最佳模型到: {best_path}")


def main(args):
    """主训练函数"""
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置设备
    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )
    print(f"使用设备: {device}")

    # 创建模型
    print(f"创建模型: {args.model_type}")
    model = create_model(
        model_type=args.model_type,
        num_classes=2,
        input_size=(args.image_size, args.image_size),
    )
    model = model.to(device)

    print(f"模型参数量: {count_parameters(model):,}")
    print(f"模型大小: {get_model_size(model):.2f} MB")

    # 数据加载
    train_transform = get_transforms(args.image_size, "train")
    val_transform = get_transforms(args.image_size, "val")

    train_dataset = DefectDataset(
        data_dir=args.data_dir,
        split="train",
        transform=train_transform,
    )

    val_dataset = DefectDataset(
        data_dir=args.data_dir,
        split="val",
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练循环
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*50}")

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # 验证
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)

        # 更新学习率
        scheduler.step()

        # 记录历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"\n训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%"
        )
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")

        # 保存检查点
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc

        checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pth"
        save_checkpoint(
            model,
            optimizer,
            epoch,
            best_acc,
            str(checkpoint_path),
            is_best=is_best,
        )

    # 保存训练历史
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n训练完成！最佳验证准确率: {best_acc:.2f}%")
    print(f"模型保存在: {output_dir}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="缺陷检测模型训练")

    # 数据参数
    parser.add_argument("--data_dir", type=str, default="data", help="数据目录")
    parser.add_argument("--output_dir", type=str, default="outputs", help="输出目录")

    # 模型参数
    parser.add_argument(
        "--model_type",
        type=str,
        default="cnn",
        choices=["cnn", "vit"],
        help="模型类型",
    )
    parser.add_argument("--image_size", type=int, default=224, help="图像尺寸")
    parser.add_argument("--pretrained", action="store_true", help="使用预训练模型")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")

    # 设备参数
    parser.add_argument("--gpu", type=int, default=0, help="GPU编号 (-1表示CPU)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
