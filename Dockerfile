# Base com CUDA (mesma que testamos)
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Básico do sistema
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git ca-certificates build-essential python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Criar ambiente venv (opcional, mas deixa o pip isolado)
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip

# Instalar libs principais (versões compatíveis com CUDA 12.4)
RUN pip install --no-cache-dir numpy pandas scikit-learn xgboost matplotlib seaborn && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

# Define o diretório de trabalho
WORKDIR /workspace

# Comando padrão ao iniciar o container
CMD ["bash"]
