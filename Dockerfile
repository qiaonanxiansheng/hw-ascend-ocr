ARG BASE_IMAGE=swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.1.rc1-310p-ubuntu22.04-py3.11
FROM ${BASE_IMAGE}

# pip 使用清华源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy project
WORKDIR /workspace
COPY . /workspace
RUN chmod +x /workspace/entrypoint.sh

# Load CANN environment in interactive shells
RUN echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> /root/.bashrc

# Expose OCR service port
EXPOSE 13502

# Entrypoint: 按需转换模型后启动服务
# 通过 -e CONVERT_MODELS=auto|always|never 控制是否转换模型(默认 auto: 缺失才转换)
ENTRYPOINT ["/bin/bash", "/workspace/entrypoint.sh"]

# Default command: start OCR API service
CMD []
