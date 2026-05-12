import torch
import torch.nn as nn

class DiffusionScheduler:
    """
    DDPM forward process scheculer.
    Compute beta, alpha, alpha_bar and use them in q_sample by indexing
    """

    def __init__(
            self,
            T: int = 1000,
            beta_start: float = 1e-4,
            beta_end: float = 0.02,
            device: str = "cpu",
    ):
        self.T = T

        # linear schedule for beta: beta_1 = 1e-4, beta_T = 0.02
        self.beta = torch.linspace(beta_start, beta_end, T, device=device)

        # precompute alpha and alpha_bar for each timestep
        self.alpha = 1. - self.beta

        # cumulative product: alpha_bar_t = alpha_1 * alpha_2 * ... * alpha_t
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1. - self.alpha_bar)
    
    def q_sample(
            self,
            x_0: torch.Tensor,
            t: torch.Tensor,
            noise: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward process: add noise to x_0 at timestep t
        
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        
        Args:
            x_0: (B, C, H, W) original image
            t: (B,) timesteps for each image in the batch(dtype=long)
            noise: (B, C, H, W) noise to add. If None, sample from standard normal.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # index t(B,) to (B, 1, 1, 1) for broadcasting
        sqrt_ab = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)

        return sqrt_ab * x_0 + sqrt_omab * noise
    
if __name__ == "__main__":
    print("=== DiffusionScheduler sanity check ===")

    scheduler = DiffusionScheduler(T=1000)

    # Schedule shape & range 확인
    print(f"beta:       shape {tuple(scheduler.beta.shape)}, "
          f"range [{scheduler.beta[0]:.2e}, {scheduler.beta[-1]:.2e}]")
    print(f"alpha_bar:  shape {tuple(scheduler.alpha_bar.shape)}, "
          f"range [{scheduler.alpha_bar[-1]:.2e}, {scheduler.alpha_bar[0]:.4f}]")

    # Dummy 이미지
    B = 4
    x_0 = torch.randn(B, 1, 28, 28)

    # ---- t=0: 거의 변화 없어야 함 ----
    t_low = torch.zeros(B, dtype=torch.long)
    x_low = scheduler.q_sample(x_0, t_low)
    diff_low = (x_low - x_0).abs().mean().item()
    print(f"\nt=0:    |x_t - x_0| 평균 = {diff_low:.4f}  (작아야 함, 보통 < 0.05)")

    # ---- t=T-1: 거의 순수 가우시안 노이즈 ----
    t_high = torch.full((B,), 999, dtype=torch.long)
    x_high = scheduler.q_sample(x_0, t_high)
    print(f"t=999:  mean = {x_high.mean().item():+.4f}, "
          f"std = {x_high.std().item():.4f}  (mean ≈ 0, std ≈ 1 이어야 함)")

    # ---- Shape 체크 ----
    assert x_low.shape == x_0.shape
    assert x_high.shape == x_0.shape
    print("\nShape OK")
