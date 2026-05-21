# check_checkpoint.py
import torch
from unet import UNet

ckpt = torch.load('checkpoints/mnist_10epoch.pt', map_location='cpu')

print(f"Saved epoch: {ckpt['epoch']}")
print(f"Final epoch loss: {ckpt['history']['epoch_losses'][-1]:.4f}")
print(f"Total iterations recorded: {len(ckpt['history']['iter_losses']):,}")

unet = UNet(
    image_channels=1, n_channels=64, ch_mults=(1, 2, 4), time_channels=64 * 4,
)
unet.load_state_dict(ckpt['model_state_dict'])
print("Checkpoint loaded successfully ✓")
print(f"Model parameters: {sum(p.numel() for p in unet.parameters()):,}")