FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.1.rc1-310p-ubuntu22.04-py3.11

# pip 使用清华源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 Python 依赖
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 拷贝项目并安装包（--no-deps：依赖已在上面装完）
WORKDIR /workspace
COPY . /workspace
RUN pip install --no-cache-dir --no-deps -e .

# 让交互式 shell 自动加载 CANN 环境变量（acl 库路径等）
RUN echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> /root/.bashrc

# 暴露 OCR 服务端口
EXPOSE 13502

# 默认命令：启动 OCR API 服务
CMD ["/bin/bash", "-c", "source /usr/local/Ascend/ascend-toolkit/set_env.sh && uvicorn api.app:app --host 0.0.0.0 --port 13502"]
