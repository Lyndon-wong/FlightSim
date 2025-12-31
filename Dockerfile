FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . /app

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 确保 gradio 6.2.0 是最终版本（覆盖任何可能的降级）
RUN pip install --no-cache-dir "gradio>=6.2.0" "huggingface_hub>=0.25.1,<1.0.0"

# 创建用户目录
RUN mkdir -p /home/user && ln -s /app /home/user/app || true

# 暴露端口
EXPOSE 7860

# 启动命令
CMD ["python", "examples/gradio_demo.py"]
