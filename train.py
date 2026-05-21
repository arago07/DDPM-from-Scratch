import torch
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt

from unet import UNet
from diffusion import DiffusionScheduler

import os
import time

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

def train(
        unet,
        scheduler,
        optimizer,
        loader,
        n_epochs,
        device='cpu',
        checkpoint_path='checkpoints/model.pt',
        log_every_n_iter=100,
        max_iters_per_epoch=None,
):
    """
    Full training loop over multiple epochs.
    
    Args:
        n_epochs: number of epochs to train
        device: 'cpu' or 'cuda'
        checkpoint_path: where to save model checkpoints
        log_every_n_iter: how often to log training loss
        max_iters_per_epoch: if None, train all iterations in each epoch
    
    Returns:
        history dict ('epoch_losses', 'iter_losses')
    
    """
    unet.train()
    history = {'epoch_losses': [], 'iter_losses': []}

    for epoch in range(n_epochs):
        epoch_start = time.time()
        epoch_loss = []

        for i, (x_0, _) in enumerate(loader):
            if max_iters_per_epoch is not None and i >= max_iters_per_epoch:
                break

            x_0 = x_0.to(device)
            loss = training_step(unet, scheduler, x_0, optimizer)
            epoch_loss.append(loss)
            history['iter_losses'].append(loss)

            if (i + 1) % log_every_n_iter == 0:
                avg_recent = sum(epoch_loss[-log_every_n_iter:]) / log_every_n_iter
                print(f"   iter {i+1:4d} | recent avg loss: {avg_recent:.4f}")

        avg_loss = sum(epoch_loss) / len(epoch_loss)
        elapsed = time.time() - epoch_start
        history['epoch_losses'].append(avg_loss)

        print(f"Epoch {epoch+1}/{n_epochs} | Avg loss: {avg_loss:.4f} | Time: {elapsed:.1f}s\n")

    # Save checkpoint
    os.makedirs(os.path.dirname(checkpoint_path) or '.', exist_ok=True)
    torch.save({
        'epoch': n_epochs,
        'model_state_dict': unet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}\n")

    return history
        
if __name__ == "__main__":
    from data import get_mnist_loader

    print("=== Full training loop test (local, 50 iter only) ===\n")

    # --- Setup ---
    T = 1000
    n_epochs = 1
    batch_size = 32
    device = 'cpu'  # Colab 에서 'cuda' 로

    unet = UNet(
        image_channels=1, n_channels=64, ch_mults=(1, 2, 4), time_channels=64 * 4,
    ).to(device)
    scheduler = DiffusionScheduler(T=T, device=device)
    optimizer = Adam(unet.parameters(), lr=1e-4)

    # MNIST loader
    loader = get_mnist_loader(batch_size=batch_size)
    print(f"Total batches available: {len(loader)}")
    print("Local test: 처음 50 iteration 만 돌려 train() 함수 검증\n")

    # --- Run training ---
    history = train(
        unet, scheduler, optimizer, loader,
        n_epochs=n_epochs,
        device=device,
        checkpoint_path='checkpoints/local_test.pt',
        log_every_n_iter=10,
        max_iters_per_epoch=50,    # 빠른 검증용
    )

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(history['iter_losses'])
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.title("Local test: 50 iterations")
    plt.grid(True, alpha=0.3)
    plt.savefig("loss_curve_local.png", dpi=100, bbox_inches="tight")
    print("\nSaved: loss_curve_local.png")