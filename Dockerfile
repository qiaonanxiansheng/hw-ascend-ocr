ARG BASE_IMAGE=cann:8.1.rc1-ubuntu22.04-py3.11
FROM ${BASE_IMAGE}

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
EXPOSE 8000

# Default command: start OCR API service
CMD ["/bin/bash", "-c", "source /usr/local/Ascend/ascend-toolkit/set_env.sh && uvicorn api.app:app --host 0.0.0.0 --port 8000"]
