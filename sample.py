import torch
import matplotlib.pyplot as plt

from unet import UNet
from diffusion import DiffusionScheduler

def main():
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = 1000
    n_samples = 16
    checkpoint_path = 'checkpoints/mnist_10epoch.pt'

    print(f"Using device: {device}")

    # Load Learned UNet model
    unet = UNet(
        image_channels=1, n_channels=64, ch_mults=[1, 2, 4], time_channels=64 * 4,  
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    unet.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    # Scheduler
    scheduler = DiffusionScheduler(T=T, device=device)

    # Generate sample images
    print(f"\nGenerating {n_samples} samples...")
    samples = scheduler.sample(
        unet, n_samples=n_samples,
        image_channels=1, image_size=28, device=device
    )

    # Denormalize: [-1, 1] -> [0, 1] for visualization
    samples = (samples.clamp(-1, 1) + 1) / 2
    samples = samples.cpu()

    # 4 x 4 grid
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i].squeeze(), cmap='gray')
        ax.axis('off')
    plt.suptitle("Generated MNIST samples (10-epoch model)")
    plt.tight_layout()
    plt.savefig("generated_samples.png", dpi=100, bbox_inches='tight')
    print("\nSaved: generated_samples.png")

if __name__ == "__main__":
    main()