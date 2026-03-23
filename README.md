# 工业AI视觉缺陷检测平台

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

基于深度学习的工业产品表面缺陷检测与智能分析平台，支持划痕、裂纹、污渍、凹坑等多种缺陷类型的自动识别。

## 功能特性

-  **图像预处理** - 自动归一化、尺寸调整、数据增强
-  **深度学习模型** - 支持CNN和ViT两种架构
-  **RESTful API** - 基于FastAPI的高性能服务接口
-  **可视化标注** - 检测结果图像标注
-  **批量检测** - 支持多张图片同时检测
-  **中文界面** - 完整的API文档中文支持

## 快速开始

### 环境要求

- Python 3.9+
- 4GB+ RAM
- 可选: NVIDIA GPU (CUDA 11.8+)

### 安装

```bash
# 克隆项目
git clone https://github.com/yourusername/defect-detection.git
cd defect-detection

# 创建虚拟环境
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 启动服务

```bash
# 方式1: Windows一键启动
run.bat

# 方式2: 手动启动
python generate_synthetic_data.py  # 生成测试数据
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

访问 http://localhost:8000/docs 查看API文档并测试。

## 项目结构

```
defect-detection/
├── app/                      # FastAPI服务
│   ├── __init__.py
│   └── main.py              # API服务主程序
├── data/                     # 数据集目录
│   ├── train/               # 训练集
│   ├── val/                 # 验证集
│   └── test/                # 测试集
├── models/                   # 模型定义
│   ├── __init__.py
│   └── defect_detector.py   # CNN/ViT模型
├── utils/                    # 工具模块
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── augmentation.py
│   └── visualization.py
├── outputs/                  # 训练输出
├── requirements.txt          # Python依赖
├── train.py                  # 训练脚本
├── generate_synthetic_data.py
├── run.bat                   # Windows启动脚本
├── Dockerfile                # Docker配置
├── README.md                 # 项目说明
├── DEPLOY.md                 # 部署指南
└── .gitignore
```

## API接口

### 单图检测

```bash
POST /predict
Content-Type: multipart/form-data

参数: file (图像文件)
```

**返回示例:**
```json
{
  "success": true,
  "result": {
    "is_defect": false,
    "defect_probability": 0.15,
    "normal_probability": 0.85,
    "predicted_class": "normal",
    "confidence": 0.85
  },
  "processing_time": 0.05,
  "visualization": "base64编码的标注图像"
}
```

### 批量检测

```bash
POST /predict_batch
Content-Type: multipart/form-data

参数: files (多张图像文件)
```

### 健康检查

```bash
GET /health
```

### 模型信息

```bash
GET /info
```

## 模型训练

### 准备数据

按以下目录结构组织数据：
```
data/
├── train/
│   ├── normal/          # 正常样本
│   └── defect/          # 缺陷样本
├── val/
│   ├── normal/
│   └── defect/
└── test/
    ├── normal/
    └── defect/
```

### 开始训练

```bash
python train.py \
    --data_dir data \
    --output_dir outputs \
    --model_type cnn \
    --epochs 30 \
    --batch_size 32 \
    --lr 0.001
```

**参数说明:**
- `--model_type`: 模型类型 (cnn/vit)
- `--epochs`: 训练轮数
- `--batch_size`: 批次大小
- `--lr`: 学习率

## 部署

### Docker部署

```bash
# 构建镜像
docker build -t defect-detection .

# 运行容器
docker run -d -p 8000:8000 --name defect-detection defect-detection
```

### 生产环境部署

详见 [DEPLOY.md](DEPLOY.md)

## 技术栈

- **深度学习**: PyTorch 2.0+, TorchVision
- **图像处理**: OpenCV, Pillow
- **Web框架**: FastAPI, Uvicorn
- **数据处理**: NumPy, Pandas, Scikit-learn
- **可视化**: Matplotlib, Seaborn

## 支持的缺陷类型

| 缺陷类型 | 描述 | 示例 |
|---------|------|------|
| 划痕 | 表面线性刮伤 | 金属表面划痕 |
| 裂纹 | 细小裂纹 | 陶瓷裂纹 |
| 污渍 | 颜色异常区域 | 油污、水渍 |
| 凹坑 | 表面变形 | 冲压凹痕 |

## 模型性能

| 模型 | 参数量 | 推理速度 | 适用场景 |
|------|--------|---------|---------|
| CNN | ~200K | 快 | 实时检测 |
| ViT | ~1.2M | 中等 | 精度优先 |

## 注意事项

1. **模型权重**: 首次启动使用随机初始化权重，生产环境需替换为训练好的模型
2. **GPU加速**: 自动检测CUDA，无需额外配置
3. **图像格式**: 支持JPEG、PNG，建议尺寸224x224
4. **内存占用**: 约500MB-1GB

## 常见问题

**Q: 如何训练自己的模型?**
A: 按上述数据格式准备数据，运行 `python train.py` 即可。

**Q: 如何更新模型?**
A: 将训练好的 `.pth` 文件放入 `outputs/` 目录，重启服务即可。

**Q: 支持GPU加速吗?**
A: 支持，自动检测CUDA环境。

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

[MIT License](LICENSE)

## 联系方式

如有问题，请提交Issue或联系维护者。
