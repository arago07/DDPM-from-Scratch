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
    
    def p_sample(self, unet, x_t, t):
        """
        Reverse process: predict x_{t-1} from x_t using the UNet model
        
        x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * pred_noise) + sigma_t * z
        
        Args:
            unet: UNet model to predict noise
            x_t: (B, C, H, W) noisy image at timestep t
            t: (B,) timesteps for each image in the batch(dtype=long)

        Returns:
            x_{t-1}: (B, C, H, W) less noisy image
        """
        B = x_t.shape[0]

        # t -> (B,) tensor
        t_batch = torch.full((B,), t, device=x_t.device, dtype=torch.long)

        # UNet으로 noise 예측
        pred_noise = unet(x_t, t_batch)

        # alpha_t, alpha_bar_t, beta_t
        alpha_t = self.alpha[t]
        alpha_bar_t = self.alpha_bar[t]
        beta_t = self.beta[t]

        # the mean of reverse distribution
        coef = beta_t / torch.sqrt(1 - alpha_bar_t)
        mean = (1 / torch.sqrt(alpha_t)) * (x_t - coef * pred_noise)

        # in the last step (t=0), we don't add noise
        if t > 0:
            z = torch.randn_like(x_t)
            sigma_t = torch.sqrt(beta_t)
            return mean + sigma_t * z
        else:
            return mean
    
    @torch.no_grad()
    def sample(self, unet, n_samples, image_channels=1, image_size=28, device="cpu"):
        """
        Generate image from pure noise by iteratively applying p_sample from t=T-1 to t=0
        
        Returns:
            x_0: generated image (n_samples, C, H, W)
        """
        unet.eval()

        # start from x_T ~ N(0, 1)
        x = torch.randn(n_samples, image_channels, image_size, image_size, device=device)

        # t = T-1, T-2, ..., 0
        for t in reversed(range(self.T)):
            x = self.p_sample(unet, x, t)

            if (t % 200 == 0):
                print(f".   sampling... t={t}")

        return x
    
    @torch.no_grad()
    def sample_progressive(self, unet, n_samples, image_channels=1, image_size=28, device="cpu", save_timesteps=None):
        """
        Generate image from pure noise by iteratively applying p_sample from t=T-1 to t=0
        Return intermediate images for visualization

        Returns:
            snapshots: dict {timestep: (n_samples, C, H, W) image at that timestep}
        """
        unet.eval()

        if save_timesteps is None:
            save_timesteps = [800, 600, 400, 200, 100, 50, 0]
        save_set = set(save_timesteps)

        # start from pure noise
        x = torch.randn(n_samples, image_channels, image_size, image_size, device=device)
        snapshots = {self.T: x.clone().cpu()} # save the initial noise as well

        for t in reversed(range(self.T)):
            x = self.p_sample(unet, x, t)

            if t in save_set:
                snapshots[t] = x.clone().cpu() # save intermediate image for visualization

        return snapshots