# syntax=docker/dockerfile:1
# =============================================================================
# Bee Tracker — imagem única com dois alvos de build: CPU (padrão) ou GPU.
#
#   CPU (padrão, build rápida):
#     docker build -t bee-tracker:cpu .
#     docker run -p 8501:8501 bee-tracker:cpu
#
#   GPU (compila OpenCV com CUDA/cuDNN — build demorada, 30-60 min):
#     docker build --target gpu -t bee-tracker:gpu .
#     docker run --gpus all -p 8501:8501 bee-tracker:gpu
#
#   Para acelerar a build GPU, restrinja CUDA_ARCH_BIN à compute capability da
#   SUA placa — descubra com: nvidia-smi --query-gpu=compute_cap --format=csv
#     docker build --target gpu --build-arg CUDA_ARCH_BIN="7.5" -t bee-tracker:gpu .
#   (GTX 10xx = 6.1, V100 = 7.0, T4/RTX 20xx = 7.5, RTX 30xx = 8.6, RTX 40xx = 8.9)
#   ATENÇÃO: valor errado faz o app cair para CPU em tempo de execução.
#
#   Requisito para GPU: driver NVIDIA no host + NVIDIA Container Toolkit.
# =============================================================================

ARG OPENCV_VERSION=4.10.0

# =============================================================================
# Estágio 1: compilação do OpenCV com CUDA (somente para o alvo GPU)
# =============================================================================
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS gpu-build
ARG OPENCV_VERSION
# Arquiteturas suportadas por padrão (Pascal -> Hopper). Restrinja p/ build mais rápida.
ARG CUDA_ARCH_BIN="6.1 7.0 7.5 8.0 8.6 8.9 9.0"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake pkg-config wget unzip \
        python3-dev python3-pip \
        libavcodec-dev libavformat-dev libswscale-dev libavutil-dev \
        libjpeg-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir numpy

RUN wget -q -O /tmp/opencv.zip \
        https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.zip \
    && wget -q -O /tmp/contrib.zip \
        https://github.com/opencv/opencv_contrib/archive/refs/tags/${OPENCV_VERSION}.zip \
    && unzip -q /tmp/opencv.zip -d /tmp \
    && unzip -q /tmp/contrib.zip -d /tmp \
    && rm /tmp/opencv.zip /tmp/contrib.zip

RUN cmake -S /tmp/opencv-${OPENCV_VERSION} -B /tmp/build \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_INSTALL_PREFIX=/opt/opencv \
        -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib-${OPENCV_VERSION}/modules \
        -D WITH_CUDA=ON \
        -D WITH_CUDNN=ON \
        -D OPENCV_DNN_CUDA=ON \
        -D CUDA_ARCH_BIN="${CUDA_ARCH_BIN}" \
        -D ENABLE_FAST_MATH=ON \
        -D CUDA_FAST_MATH=ON \
        -D BUILD_opencv_python3=ON \
        -D OPENCV_PYTHON3_INSTALL_PATH=/opt/opencv/python \
        -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
        -D WITH_FFMPEG=ON \
        -D WITH_GTK=OFF \
        -D WITH_QT=OFF \
        -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D BUILD_EXAMPLES=OFF \
        -D BUILD_opencv_apps=OFF \
        -D BUILD_JAVA=OFF \
    && cmake --build /tmp/build -j"$(nproc)" \
    && cmake --install /tmp/build \
    && rm -rf /tmp/build /tmp/opencv-${OPENCV_VERSION} /tmp/opencv_contrib-${OPENCV_VERSION}

# =============================================================================
# Alvo GPU: runtime CUDA enxuto + OpenCV compilado no estágio anterior
# =============================================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS gpu
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=gpu-build /opt/opencv /opt/opencv
ENV PYTHONPATH=/opt/opencv/python \
    LD_LIBRARY_PATH=/opt/opencv/lib:${LD_LIBRARY_PATH}

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

# Banco SQLite e logs — monte volumes nesses caminhos para persistir os dados
RUN mkdir -p /app/data /app/logs
ENV BEE_DB_PATH=/app/data/bee_tracker.db \
    BEE_LOG_DIR=/app/logs

ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_MAX_UPLOAD_SIZE=3000

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "src/bee_tracker.py"]

# =============================================================================
# Alvo CPU (padrão — último estágio): leve, usa opencv-python-headless do pip
# =============================================================================
FROM python:3.11-slim AS cpu
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt opencv-python-headless

COPY src/ ./src/
COPY models/ ./models/

# Banco SQLite e logs — monte volumes nesses caminhos para persistir os dados
RUN mkdir -p /app/data /app/logs
ENV BEE_DB_PATH=/app/data/bee_tracker.db \
    BEE_LOG_DIR=/app/logs

ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_MAX_UPLOAD_SIZE=3000

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "src/bee_tracker.py"]
