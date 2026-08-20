FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.1.rc1-310p-ubuntu22.04-py3.11

# pip 使用清华源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 Python 依赖（容器内无 GUI，opencv 使用 headless 版本，避免 libGL 依赖问题）
RUN pip install --no-cache-dir \
        "numpy>=1.21.0" \
        "opencv-python-headless>=4.5.0" \
        "pyclipper>=1.2.0" \
        pytest

# 拷贝项目并安装包（--no-deps：依赖已在上面装完）
WORKDIR /workspace
COPY . /workspace
RUN pip install --no-cache-dir --no-deps -e .

# 让交互式 shell 自动加载 CANN 环境变量（acl 库路径等）
RUN echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> /root/.bashrc

# 默认命令：运行单元测试（交互式运行时会被 /bin/bash 覆盖）
CMD ["/bin/bash", "-c", "source /usr/local/Ascend/ascend-toolkit/set_env.sh && python -m pytest tests/ -v"]
