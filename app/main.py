# -*- coding: utf-8 -*-
"""
FastAPI 缺陷检测服务

提供REST API接口，支持图像上传、缺陷检测、结果可视化。
"""

import io
import base64
import tempfile
from pathlib import Path
from typing import Optional, List
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import yaml

# 项目导入
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import create_model, DefectDetector
from utils import ImagePreprocessor


# ============ 配置 ============

CONFIG = {
    "model_path": "outputs/best_model.pth",
    "model_type": "cnn",
    "image_size": 224,
    "threshold": 0.5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "classes": ["normal", "defect"],
}


# ============ FastAPI 应用 ============

app = FastAPI(
    title="工业缺陷检测系统",
    description="基于深度学习的工业产品表面缺陷检测与智能分析平台",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ 全局变量 ============

model = None
detector = None
preprocessor = None


# ============ 数据模型 ============

class PredictionResult(BaseModel):
    """预测结果"""
    is_defect: bool = Field(..., description="是否存在缺陷", title="是否缺陷")
    defect_probability: float = Field(..., description="缺陷概率(0-1)", title="缺陷概率")
    normal_probability: float = Field(..., description="正常概率(0-1)", title="正常概率")
    predicted_class: str = Field(..., description="预测类别(normal/defect)", title="预测类别")
    confidence: float = Field(..., description="置信度", title="置信度")

    class Config:
        json_schema_extra = {
            "example": {
                "is_defect": False,
                "defect_probability": 0.15,
                "normal_probability": 0.85,
                "predicted_class": "normal",
                "confidence": 0.85
            }
        }


class DetectionResponse(BaseModel):
    """检测响应"""
    success: bool = Field(..., description="是否成功", title="成功状态")
    message: str = Field(..., description="响应消息", title="消息")
    result: Optional[PredictionResult] = Field(None, description="检测结果")
    processing_time: Optional[float] = Field(None, description="处理时间(秒)", title="处理时间")
    visualization: Optional[str] = Field(None, description="Base64编码的可视化图像", title="可视化图像")


# ============ 工具函数 ============

def load_model():
    """加载模型"""
    global model, detector, preprocessor

    print("正在加载模型...")

    # 创建模型
    model = create_model(
        model_type=CONFIG["model_type"],
        num_classes=2,
        pretrained=False,
    )

    # 尝试加载预训练权重
    model_path = Path(project_root) / CONFIG["model_path"]
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=CONFIG["device"])
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"已加载模型权重: {model_path}")
    else:
        print(f"未找到模型文件: {model_path}，使用随机初始化权重")

    # 创建检测器
    detector = DefectDetector(
        model=model,
        device=CONFIG["device"],
        threshold=CONFIG["threshold"],
    )

    # 创建预处理器
    preprocessor = ImagePreprocessor(
        target_size=(CONFIG["image_size"], CONFIG["image_size"]),
        normalize=True,
    )

    print(f"模型加载完成，使用设备: {CONFIG['device']}")


def process_image(image: Image.Image) -> tuple:
    """
    处理图像并进行预测

    Args:
        image: PIL图像对象

    Returns:
        (预测结果, 可视化图像numpy数组)
    """
    # 转换为numpy数组 (BGR)
    image_np = np.array(image.convert("RGB"))
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # 预处理
    image_preprocessed = preprocessor.preprocess(image_bgr)
    image_tensor = torch.from_numpy(image_preprocessed).unsqueeze(0)

    # 预测
    result = detector.predict(image_tensor, return_probs=True)

    # 提取结果
    is_defect = result["is_defect"]
    defect_prob = result["defect_probability"]
    normal_prob = 1.0 - defect_prob
    predicted_class = CONFIG["classes"][result["class"]]
    confidence = max(defect_prob, normal_prob)

    # 生成可视化图像
    vis_image = visualize_detection(image_bgr, defect_prob, CONFIG["threshold"])

    # 转换为base64
    _, buffer = cv2.imencode(".jpg", vis_image)
    vis_base64 = base64.b64encode(buffer).decode("utf-8")

    prediction = PredictionResult(
        is_defect=is_defect,
        defect_probability=defect_prob,
        normal_probability=normal_prob,
        predicted_class=predicted_class,
        confidence=confidence,
    )

    return prediction, vis_base64


def visualize_detection(
    image: np.ndarray,
    defect_prob: float,
    threshold: float,
) -> np.ndarray:
    """
    生成检测结果可视化图像

    Args:
        image: 原始图像 (BGR)
        defect_prob: 缺陷概率
        threshold: 判定阈值

    Returns:
        可视化图像
    """
    result = image.copy()
    h, w = image.shape[:2]

    # 绘制状态栏
    status_bar_height = 60
    status_bar = np.zeros((status_bar_height, w, 3), dtype=np.uint8)

    is_defect = defect_prob > threshold

    # 设置颜色 (BGR格式)
    if is_defect:
        status_color = (0, 0, 255)  # 红色 - 有缺陷
        status_text = "检测到缺陷"
    else:
        status_color = (0, 255, 0)  # 绿色 - 正常
        status_text = "正常"

    # 绘制状态背景
    cv2.rectangle(status_bar, (0, 0), (w, status_bar_height), status_color, -1)

    # 添加文字
    cv2.putText(
        status_bar,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2,
    )

    # 添加概率信息
    prob_text = f"缺陷概率: {defect_prob:.2%}"
    text_size = cv2.getTextSize(prob_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.putText(
        status_bar,
        prob_text,
        (w - text_size[0] - 20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # 合并图像
    result = np.vstack([status_bar, result])

    return result


def create_demo_prediction():
    """创建演示用预测结果（当模型未加载时）"""
    return PredictionResult(
        is_defect=False,
        defect_probability=0.15,
        normal_probability=0.85,
        predicted_class="normal",
        confidence=0.85,
    )


# ============ API 端点 ============

@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    try:
        load_model()
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("将使用演示模式运行")


@app.get("/", summary="服务状态", description="获取服务运行状态和基本信息")
async def root():
    """获取服务状态"""
    return {
        "服务名称": "工业缺陷检测系统",
        "版本": "1.0.0",
        "状态": "运行中",
        "模型已加载": detector is not None,
    }


@app.get("/health", summary="健康检查", description="检查服务健康状态和模型加载情况")
async def health_check():
    """健康检查"""
    return {
        "状态": "健康",
        "模型已加载": detector is not None,
        "运行设备": CONFIG["device"],
    }


@app.post(
    "/predict",
    response_model=DetectionResponse,
    summary="单图缺陷检测",
    description="上传单张图像进行缺陷检测，返回检测结果和可视化图像"
)
async def predict(file: UploadFile = File(..., description="要检测的图像文件(JPEG/PNG)")):
    """
    单图缺陷检测接口

    上传一张产品表面图像，系统自动分析是否存在缺陷。

    **参数:**
    - file: 图像文件(JPEG/PNG格式)

    **返回:**
    - success: 是否检测成功
    - result: 检测结果
        - is_defect: 是否存在缺陷(true/false)
        - defect_probability: 缺陷概率(0-1)
        - predicted_class: 预测类别(normal/defect)
        - confidence: 置信度
    - processing_time: 处理耗时(秒)
    - visualization: 标注结果图像(Base64)
    """
    import time
    start_time = time.time()

    try:
        # 读取图像
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # 检查图像格式
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        # 处理图像
        if detector is not None:
            prediction, vis_base64 = process_image(image)
        else:
            # 演示模式
            prediction = create_demo_prediction()
            vis_base64 = None

        processing_time = time.time() - start_time

        return DetectionResponse(
            success=True,
            message="检测成功",
            result=prediction,
            processing_time=processing_time,
            visualization=vis_base64,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理图像时出错: {str(e)}")


@app.post(
    "/predict_batch",
    response_model=List[DetectionResponse],
    summary="批量缺陷检测",
    description="同时上传多张图像进行批量缺陷检测"
)
async def predict_batch(files: List[UploadFile] = File(default=None, description="要检测的图像文件列表")):
    """
    批量缺陷检测接口

    同时上传多张产品表面图像，批量检测是否存在缺陷。

    **参数:**
    - files: 图像文件列表

    **返回:**
    每张图像的检测结果列表
    """
    import time
    results = []

    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))

            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")

            if detector is not None:
                prediction, _ = process_image(image)
            else:
                prediction = create_demo_prediction()

            results.append(
                DetectionResponse(
                    success=True,
                    message="检测成功",
                    result=prediction,
                )
            )

        except Exception as e:
            results.append(
                DetectionResponse(
                    success=False,
                    message=f"处理失败: {str(e)}",
                )
            )

    return results


@app.get("/info", summary="模型信息", description="获取当前使用的模型配置信息")
async def get_model_info():
    """获取模型信息"""
    return {
        "模型类型": CONFIG["model_type"],
        "输入图像尺寸": f"{CONFIG['image_size']}x{CONFIG['image_size']}",
        "检测阈值": CONFIG["threshold"],
        "支持类别": ["正常(normal)", "缺陷(defect)"],
        "运行设备": CONFIG["device"],
    }


# ============ 主程序 ============

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
