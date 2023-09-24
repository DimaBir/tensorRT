# TensorRT and ONNX Example Project

This project fine-tunes DistilBERT and compares inference without using TensorRT and with TensorRT on a model converted from PyTorch to ONNX.

## Setup

1. **Build the Docker image:**
```sh
docker build -t tensorrt_project -f docker/Dockerfile .
