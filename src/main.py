from model import DistilBERTModel
import time

def main():
    model = DistilBERTModel()

    question = "What is the capital of France?"
    context = "Paris is the capital of France."

    inputs = model.tokenize(question, context)

    # Infer with PyTorch
    start_time = time.time()
    output_pytorch = model.infer_pytorch(inputs)
    print("PyTorch Inference Time:", time.time() - start_time)

    # Convert to ONNX
    onnx_path = "distilbert.onnx"
    model.convert_to_onnx(onnx_path, inputs)

    # Infer with TensorRT FP32
    engine_fp32 = model.build_engine(onnx_path, fp16_mode=False)
    start_time = time.time()
    output_trt_fp32 = model.infer_tensorrt(engine_fp32, inputs)
    print("TensorRT FP32 Inference Time:", time.time() - start_time)

    # Infer with TensorRT FP16
    engine_fp16 = model.build_engine(onnx_path, fp16_mode=True)
    start_time = time.time()
    output_trt_fp16 = model.infer_tensorrt(engine_fp16, inputs)
    print("TensorRT FP16 Inference Time:", time.time() - start_time)

    # Print the outputs
    print("PyTorch Output:", output_pytorch)
    print("TensorRT FP32 Output:", output_trt_fp32)
    print("TensorRT FP16 Output:", output_trt_fp16)


if __name__ == "__main__":
    main()
