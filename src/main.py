import numpy as np
import logging
import time

from model import DistilBERTModel


def logits_to_answer(start_logits, end_logits, input_ids, tokenizer):
    print("Start Logits:", start_logits)  # Debugging line
    print("End Logits:", end_logits)  # Debugging line

    start_idx = np.argmax(start_logits)
    end_idx = np.argmax(end_logits)

    print("Start Index:", start_idx)  # Debugging line
    print("End Index:", end_idx)  # Debugging line

    # Ensure start_idx is not greater than end_idx
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    # Extract the answer tokens and filter out special tokens and punctuation
    answer_tokens = tokenizer.convert_ids_to_tokens(input_ids[start_idx:end_idx + 1])
    answer_tokens = [token for token in answer_tokens if token not in ['[CLS]', '[SEP]'] and token.isalnum()]

    print("Answer Tokens:", answer_tokens)  # Debugging line

    # If no answer tokens are left after filtering, return a default message
    if not answer_tokens:
        return "No answer found."

    # Convert tokens to string
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    return answer




def main():
    # Set up logging
    logging.basicConfig(filename='model.log', level=logging.INFO)

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
    logging.info(f"PyTorch Inference Time: {time.time() - start_time}")

    # Convert PyTorch output to answer
    pytorch_answer = logits_to_answer(
        output_pytorch.start_logits.squeeze().cpu().detach().numpy(),
        output_pytorch.end_logits.squeeze().cpu().detach().numpy(),
        inputs['input_ids'].squeeze().tolist(),
        model.tokenizer
    )
    print(f"PyTorch Answer: {pytorch_answer}")

    # Convert the PyTorch model to ONNX format
    onnx_path = "distilbert.onnx"
    model.convert_to_onnx(onnx_path, inputs)

    # Build a TensorRT engine with FP32 precision and perform inference
    engine_fp32 = model.build_engine(onnx_path, fp16_mode=False)
    start_time = time.time()
    start_logits_trt_fp32, end_logits_trt_fp32 = model.infer_tensorrt(engine_fp32, inputs)
    logging.info(f"TensorRT FP32 Inference Time: {time.time() - start_time}")

    # Convert TensorRT FP32 output to answer
    trt_fp32_answer = logits_to_answer(start_logits_trt_fp32, end_logits_trt_fp32,
                                       inputs['input_ids'].squeeze().tolist(),
                                       model.tokenizer)
    print(f"TensorRT FP32 Answer: {trt_fp32_answer}")

    # Build a TensorRT engine with FP16 precision and perform inference
    engine_fp16 = model.build_engine(onnx_path, fp16_mode=True)
    start_time = time.time()
    start_logits_trt_fp16, end_logits_trt_fp16 = model.infer_tensorrt(engine_fp16, inputs)
    logging.info(f"TensorRT FP16 Inference Time: {time.time() - start_time}")

    # Convert TensorRT FP16 output to answer
    trt_fp16_answer = logits_to_answer(start_logits_trt_fp16, end_logits_trt_fp16,
                                       inputs['input_ids'].squeeze().tolist(),
                                       model.tokenizer)
    print(f"TensorRT FP16 Answer: {trt_fp16_answer}")


if __name__ == "__main__":
    main()
