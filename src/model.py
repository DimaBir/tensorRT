import torch
import transformers
import onnx
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Necessary for using CUDA driver


class DistilBERTModel:
    def __init__(self, model_name='distilbert-base-uncased'):
        # Load the tokenizer and model
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained(model_name)
        self.model = transformers.AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.trt_engine = None

    def tokenize(self, question, context):
        # Tokenize the input question and context
        return self.tokenizer(question, context, return_tensors="pt")

    def infer_pytorch(self, inputs):
        # Perform inference using the PyTorch model
        with torch.no_grad():
            return self.model(**inputs)

    def convert_to_onnx(self, onnx_path, inputs):
        # Convert the PyTorch model to ONNX format
        torch.onnx.export(self.model, (inputs["input_ids"], inputs["attention_mask"]), onnx_path, verbose=True, opset_version=11, input_names=['input_ids', 'attention_mask'], output_names=['output'])

    def build_engine(self, onnx_path, fp16_mode=False):
        # Build a TensorRT engine from the ONNX model
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.fp16_mode = fp16_mode  # Set precision mode
            builder.max_workspace_size = 1 << 28  # Set workspace size
            builder.max_batch_size = 1  # Set batch size

            # Parse the ONNX model to create a TensorRT network
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None

            # Build and return the TensorRT engine
            return builder.build_cuda_engine(network)

    def infer_tensorrt(self, engine, inputs):
        # Perform inference using the TensorRT engine
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        context = engine.create_execution_context()

        # Allocate device memory for inputs and outputs
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(cuda_mem))
            if engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': cuda_mem})
            else:
                outputs.append({'host': host_mem, 'device': cuda_mem})

        # Transfer input data to the GPU
        cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
        cuda.memcpy_htod_async(inputs[1]['device'], inputs[1]['host'], stream)

        # Run inference
        stream.synchronize()
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
        stream.synchronize()

        # Return the output data
        return outputs[0]['host']
