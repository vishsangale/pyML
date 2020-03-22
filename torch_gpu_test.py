import torch
import torchvision

print(torch.__version__)

print(torchvision.__version__)

print(
    torch.cuda.current_device(),
    torch.cuda.device_count(),
    torch.cuda.get_device_name(0),
    torch.cuda.is_available(),
)
