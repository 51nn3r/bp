import torch

from gm.flash_attention.XformersFlashMHA import XformersFlashMHA

device = torch.device("cuda")
model = XformersFlashMHA(embed_dim=512, num_heads=8, dropout=0.1, causal=False)
model.to(device)
x = torch.randn(2, 128, 512)
x = x.to(device)
out = model(x)
print("Output shape:", out.shape)
