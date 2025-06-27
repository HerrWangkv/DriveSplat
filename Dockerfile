# Use Python 3.10 with CUDA 12.1 base image
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python (python3 already exists)
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install -r requirements.txt

# Install PyTorch3D from GitHub
RUN pip install git+https://github.com/facebookresearch/pytorch3d.git

# Remove the copied requirements.txt file
RUN rm requirements.txt

# Install squashfuse
RUN apt-get update && apt-get install -y squashfuse && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application
COPY . .

# Set default command
CMD ["/bin/bash"]