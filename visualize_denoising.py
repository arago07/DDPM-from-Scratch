import torch
import matplotlib.pyplot as plt

from unet import UNet
from diffusion import DiffusionScheduler

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = 1000
    n_samples = 4
    checkpoint_path = "checkpoints/mnist_10epoch.pt"

    print(f"Using device: {device}")

    # Initialize UNet and load checkpoint
    unet = UNet(image_channels=1, n_channels=64, ch_mults=(1, 2, 4), time_channels=64*4,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    unet.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded UNet checkpoint from epoch {ckpt['epoch']}")

    scheduler = DiffusionScheduler(T=T, device=device)

    # generate with trajectory
    save_timesteps = [800, 600, 400, 200, 100, 50, 0]
    print("\nGenerating samples with trajectory...")
    snapshots = scheduler.sample_progressive(
        unet=unet,
        n_samples=n_samples,
        device=device,
        save_timesteps=save_timesteps
    )
    timesteps = sorted(snapshots.keys(), reverse=True)
    n_cols = len(timesteps)

    fig, axes = plt.subplots(n_samples, n_cols, figsize=(1.4*n_cols, 1.4*n_samples))
    for row in range(n_samples):
        for col, t in enumerate(timesteps):
            img = (snapshots[t][row].squeeze().clamp(-1, 1) + 1) / 2
            ax = axes[row, col]
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            if row == 0:
                label = "noise" if t == T else f"t={t}"
                ax.set_title(label, fontsize=10)

    plt.suptitle("DDPM denoising: pure noise → MNIST digit (left → right)", fontsize=12)
    plt.tight_layout()
    plt.savefig("denoising_process.png", dpi=120, bbox_inches='tight')
    print("\nSaved: denoising_process.png")

if __name__ == "__main__":
    main()