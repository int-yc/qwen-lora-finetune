import torch
print("torch:", torch.__version__)
print("CUDA 可用:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
print("显存(GB):", round(gb, 1))
