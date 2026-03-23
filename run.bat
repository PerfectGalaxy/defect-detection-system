@echo off
chcp 65001 >nul
echo ========================================
echo   工业缺陷检测平台 - 一键启动脚本
echo ========================================
echo.

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.9+
    pause
    exit /b 1
)

echo [1/4] 检查并安装依赖...
pip install -r requirements.txt -q
if errorlevel 1 (
    echo [错误] 依赖安装失败
    pause
    exit /b 1
)
echo      依赖安装完成

echo.
echo [2/4] 生成合成测试数据...
python generate_synthetic_data.py
if errorlevel 1 (
    echo [警告] 数据生成失败，将跳过此步骤
)

echo.
echo [3/4] 启动FastAPI服务...
echo      服务将在 http://localhost:8000 启动
echo      API文档: http://localhost:8000/docs
echo      按 Ctrl+C 停止服务
echo.

REM 启动服务
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
