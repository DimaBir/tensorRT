from model import DistilBERTModel
import time

def main():
    # Initialize the model
    model = DistilBERTModel()

    # Define a sample input
    question = "What is the capital of France?"
    context = "Paris is the capital of France."

    # Tokenize the input
    inputs = model.tokenize(question, context)

    # Perform inference using PyTorch and measure the time taken
    start_time = time.time()
    output_pytorch = model.infer_pytorch(inputs)
    print("PyTorch Inference Time:", time.time() - start_time)

    # Convert the PyTorch model to ONNX format
    onnx_path = "distilbert.onnx"
    model.convert_to_onnx(onnx_path, inputs)

    # Build a TensorRT engine with FP32 precision and perform inference
    engine_fp32 = model.build_engine(onnx_path, fp16_mode=False)
    start_time = time.time()
    output_trt_fp32 = model.infer_tensorrt(engine_fp32, inputs)
    print("TensorRT FP32 Inference Time:", time.time() - start_time)

    # Build a TensorRT engine with FP16 precision and perform inference
    engine_fp16 = model.build_engine(onnx_path, fp16_mode=True)
    start_time = time.time()
    output_trt_fp16 = model.infer_tensorrt(engine_fp16, inputs)
    print("TensorRT FP16 Inference Time:", time.time() - start_time)

    # Print the outputs from different inference methods for comparison
    print("PyTorch Output:", output_pytorch)
    print("TensorRT FP32 Output:", output_trt_fp32)
    print("TensorRT FP16 Output:", output_trt_fp16)


if __name__ == "__main__":
    main()
