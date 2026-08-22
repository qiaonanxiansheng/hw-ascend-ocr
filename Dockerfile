FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.1.rc1-310p-ubuntu22.04-py3.11

# pip 使用清华源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy project and install package (--no-deps: deps already installed above)
WORKDIR /workspace
COPY . /workspace
RUN pip install --no-cache-dir --no-deps -e .

# Load CANN environment in interactive shells
RUN echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> /root/.bashrc

# Expose OCR service port
EXPOSE 13502

# Default command: start OCR API service
CMD ["/bin/bash", "-c", "source /usr/local/Ascend/ascend-toolkit/set_env.sh && uvicorn api.app:app --host 0.0.0.0 --port 13502"]
