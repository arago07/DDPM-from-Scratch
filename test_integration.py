import torch
from unet import UNet
from diffusion import DiffusionScheduler

def test_forward_pass():
    """
    Test if UNet + DiffusionScheduler can work together without errors.
    """
    print("=== Integration test: UNet + DiffusionScheduler ===\n")

    # Setup
    B = 4  # batch size
    T = 1000  # total diffusion steps

    unet = UNet(
        image_channels=1,
        n_channels=64,
        ch_mults=(1, 2, 4),
        time_channels=64 * 4,
    )
    scheduler = DiffusionScheduler(T=T)

    # Step 1: Create dummy input
    x_0 = torch.randn(B, 1, 28, 28)  # (B, C, H, W)
    print(f"Input x_0 shape: {tuple(x_0.shape)}")

    # Step 2: Random timestep + noise
    t = torch.randint(0, T, (B,)) # automatically long dtype
    noise = torch.randn_like(x_0)
    print(f"Random timesteps t: {tuple(t.shape)}, values: {t.tolist()}")
    print(f"Noise shape: {tuple(noise.shape)}")

    # Step 3: Forward process: add noise to x_0 at timestep t
    x_t = scheduler.q_sample(x_0, t, noise=noise)
    print(f"Noised x_t shape: {tuple(x_t.shape)}")

    # Step 4: UNet predict noise
    with torch.no_grad():
        pred_noise = unet(x_t, t)
    print(f"Predicted noise shape: {tuple(pred_noise.shape)}")

    # --- Checks ---
    print("\n--- Checks ---")
    assert pred_noise.shape == noise.shape, \
        f"Shape mismatch: pred {pred_noise.shape} vs noise {noise.shape}"
    print("pred_noise.shape == noise.shape (학습 가능한 형태) ✓")

    assert not torch.isnan(x_t).any(), "x_t contains NaN"
    assert not torch.isnan(pred_noise).any(), "pred_noise contains NaN"
    print("No NaN in x_t, pred_noise ✓")

    print(f"\npred_noise stats: mean={pred_noise.mean():+.4f}, std={pred_noise.std():.4f}")
    print("(학습 안 된 UNet 이라 의미 없는 값. shape 만 맞으면 OK)")

if __name__ == "__main__":
    test_forward_pass()
