import torch
import transformers
import onnx
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class DistilBERTModel:
    def __init__(self, model_name='distilbert-base-uncased'):
        # Load the tokenizer and model
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained(model_name)
        self.model = transformers.AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.trt_engine = None
        self.model_name = model_name

    def get_model_name(self):
        return self.model_name

    def tokenize(self, question, context):
        # Tokenize the input question and context
        return self.tokenizer(question, context, return_tensors="pt")

    def infer_pytorch(self, inputs):
        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move model to the appropriate device
        self.model.to(device)

        # Move inputs to the appropriate device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Perform inference
        with torch.no_grad():
            output = self.model(**inputs)

        return output

    def convert_to_onnx(self, onnx_path, inputs):
        # Move inputs to the same device as the model
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

        # Export the model to ONNX
        torch.onnx.export(self.model,
                          (inputs["input_ids"], inputs["attention_mask"]),
                          onnx_path,
                          verbose=True,
                          opset_version=11,
                          input_names=['input_ids', 'attention_mask'],
                          output_names=['output'])

    def build_engine(self, onnx_file_path, fp16_mode=False):
        # Create a TensorRT logger
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        # Create a builder
        builder = trt.Builder(TRT_LOGGER)

        # Create a network
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        # Create a parser
        parser = trt.OnnxParser(network, TRT_LOGGER)

        # Parse the ONNX model
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # Create a builder configuration object
        config = builder.create_builder_config()

        # Set the precision mode in the configuration
        if fp16_mode and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # Build and return the engine
        return builder.build_engine(network, config)

    def infer_tensorrt(self, engine, inputs):
        # Perform inference using the TensorRT engine
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        if engine is None:
            print("Error: Engine could not be created.")
            return

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
