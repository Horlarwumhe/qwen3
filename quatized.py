import mlx.core as mx

import sys
path = sys.argv[-1]
tensors = mx.load(path,format='safetensors')
bits = 4
quantized_tensors = {}
keys = list(tensors.keys())
for name in keys:
    tensor = tensors.pop(name)

    print(f"quantizing {name} {tensor.shape} {tensor.dtype}")
    if tensor.dtype != mx.float16:
        tensor = tensor.astype(mx.float16)
    if tensor.ndim > 1:
        quantized, scales, biases = mx.quantize(tensor,bits=bits)
        
        quantized_tensors[f"{name}"] = quantized
        quantized_tensors[f"{name}.scale"] = scales
        quantized_tensors[f"{name}.bias"] = biases
    else:
        quantized_tensors[name] = tensor


name = f"{path}-{bits}bit"
print("......saving ",name)
mx.save_safetensors(name, quantized_tensors)
