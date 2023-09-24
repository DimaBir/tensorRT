# Use an official TensorRT base image
FROM nvcr.io/nvidia/tensorrt:latest

# Install system packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    git

# Install Python packages
RUN pip3 install torch torchvision torchaudio transformers onnx onnxruntime pycuda

# Set the working directory
WORKDIR /workspace