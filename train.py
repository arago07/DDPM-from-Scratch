import torch
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt

from unet import UNet
from diffusion import DiffusionScheduler

def training_step(unet, scheduler, x_0, optimizer):
    """
    An iteration step of DDPM training.
    
    Pipeline:
        1. Each sample in the batch, sample a different random t step
        2. Sample noise from standard normal distribution
        3. Produce x_t using q_sample
        4. Predict noise using UNet: noise_pred = unet(x_t, t)
        5. MSE loss between noise_pred and the true noise (= loss)
        6. Backward + optimizer.step()

        Returns:
            loss value(float, .item()으로 tensor에서 꺼내기)
        """
    
    B = x_0.shape[0]
    T = scheduler.T

    # 1. random timestep t for each sample in the batch
    t = torch.randint(0, T, (B,), device=x_0.device)

    # 2. random noise
    noise = torch.randn_like(x_0)

    # 3. Forward process: produce x_t
    x_t = scheduler.q_sample(x_0, t, noise=noise)

    # 4. UNet predict noise
    pred_noise = unet(x_t, t)

    # 5. MSE loss
    loss = F.mse_loss(pred_noise, noise)

    # 6. Backward + optimizer step
    optimizer.zero_grad()
    loss.backward() # calculate gradients
    optimizer.step() # update parameters

    return loss.item()

if __name__ == "__main__":
    print("=== Training step smoke test (dummy data) ===\n")

    # --- Setup ---
    B = 4
    T = 1000
    n_iterations = 100

    unet = UNet(
        image_channels=1, n_channels=64, ch_mults=(1, 2, 4), time_channels=64 * 4,
    )
    scheduler = DiffusionScheduler(T=T)
    optimizer = Adam(unet.parameters(), lr=1e-4)

    # 같은 dummy 배치를 반복 학습 → loss 가 줄어들면 학습 메커니즘 작동
    x_0 = torch.randn(B, 1, 28, 28)

    # --- Training loop ---
    losses = []
    for i in range(n_iterations):
        loss = training_step(unet, scheduler, x_0, optimizer)
        losses.append(loss)

        if (i + 1) % 10 == 0:
            print(f"Iter {i+1:3d} | loss = {loss:.4f}")

    print(f"\nInitial loss: {losses[0]:.4f}")
    print(f"Final loss:   {losses[-1]:.4f}")
    print(f"Reduction:    {losses[0] - losses[-1]:+.4f}")

    # --- Loss curve plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.title("Training step smoke test (dummy data)")
    plt.grid(True, alpha=0.3)
    plt.savefig("loss_curve_dummy.png", dpi=100, bbox_inches="tight")
    print("\nSaved: loss_curve_dummy.png")