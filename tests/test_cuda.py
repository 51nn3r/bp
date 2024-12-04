import torch

if torch.cuda.is_available():
    print("GPU is available:", torch.cuda.get_device_name(0))
else:
    print("GPU is not available")

print(torch.version.cuda)  # Должно вернуть номер версии CUDA
print(torch.backends.cudnn.enabled)  # Должно вернуть True, если CUDNN активен
print(torch.cuda.is_available())  # Должно вернуть True
