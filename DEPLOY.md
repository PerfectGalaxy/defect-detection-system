# 部署指南

## 系统要求

- Python 3.9+
- 4GB+ RAM
- 可选: NVIDIA GPU (CUDA 11.8+)

## 快速部署

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/defect-detection.git
cd defect-detection
```

### 2. 创建虚拟环境

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 启动服务

```bash
# 方式1: 使用启动脚本 (Windows)
run.bat

# 方式2: 手动启动
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5. 访问服务

- API文档: http://localhost:8000/docs
- 服务状态: http://localhost:8000

## Docker部署

### 构建镜像

```bash
docker build -t defect-detection .
```

### 运行容器

```bash
docker run -d -p 8000:8000 --name defect-detection defect-detection
```

## 生产环境部署

### 使用Gunicorn + Uvicorn

```bash
pip install gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### 使用Nginx反向代理

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 注意事项

1. **模型权重**: 首次启动会自动创建随机权重模型，生产环境需替换为训练好的模型
2. **GPU加速**: 自动检测CUDA，无需额外配置
3. **内存占用**: 约500MB-1GB，批量检测时可能增加
4. **图像格式**: 支持JPEG、PNG，建议尺寸224x224

## 常见问题

**Q: 启动时报错"ModuleNotFoundError"?**
A: 确保已激活虚拟环境并安装所有依赖

**Q: 如何训练自己的模型?**
A: 参考 `python train.py --help` 查看训练参数

**Q: 如何更新模型?**
A: 将训练好的 `.pth` 文件放入 `outputs/` 目录，重启服务即可
