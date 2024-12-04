import torch


def move_to_device(
        data,
        device: torch._C.device,
):
    if isinstance(data, dict):
        return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in data.items()}
    elif isinstance(data, list):
        return [move_to_device(x, device) for x in data]
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        raise TypeError("Unsupported data type")
