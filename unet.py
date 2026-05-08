import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):

    """
    Sinusoidal time embedding + 2-layer MLP.

    Converts a scalar timestep t (an integer) into a high-dimensional
    vector that the U-Net can use to condition each ResidualBlock.

    Inspired by Transformer's positional encoding.
    """

    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(n_channels // 4, n_channels)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(n_channels, n_channels)

    def forward(self, t):
        # 1. half_dim calculation
        half_dim = self.n_channels // 8

        # 2. Create the sinusoidal embedding
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)

        # 3. Bradcast t and multiply by the embedding
        emb = t[:, None] * emb[None, :]

        # 4. sin / cos concat
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # 5. 2 MLP layers
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        return emb

class ResidualBlock(nn.Module):
    """
    Residual block with time conditioning
    
    Inputs:
        x: image features(B, in_channels, H, W)
        t_emb: time embedding (B, time_emb_dim)

    Outputs:
        (B, out_channels, H, W)
    """

    def __init__(self, in_channels:int, out_channels:int, time_emb_dim: int,
                 n_groups: int=32):
        super().__init__()
        # First conv layer
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Time embedding projection
        self.time_act = nn.SiLU()
        self.time_emb = nn.Linear(time_emb_dim, out_channels)

        # Second conv layer
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Skip connection (1X1 conv if channel size changes)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity() # meaning?

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # 1. First conv layer
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        # 2. Add time embedding
        t = self.time_act(self.time_emb(t))
        t = t[:, :, None, None]  # reshape to (B, out_channels, 1, 1)
        h = h + t

        # 3. Second conv layer
        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        # 4. Skip connection
        return h + self.shortcut(x)
    
class Downsample(nn.Module):
    """
    Downsample by factor of 2 using strided convolution
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        return self.conv(x)
        
class Upsample(nn.Module):
    """
    Upsample by factor of 2 using transposed convolution.
    """
    
    def  __init__(self, n_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        return self.conv(x)
    
class MiddleBlock(nn.Module):
    """
    Bottlneck of the U-net: 2 residual blocks
    """

    def __init__(self, n_channels: int, time_emb_dim: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_emb_dim)
        self.res2 = ResidualBlock(n_channels, n_channels, time_emb_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.res2(x, t)
        return x
    
class UNet(nn.Module):
    """
    U-Net with time condidtioning for DDPM.
    
    Architecture:
        Input -> Initial conv -> Downsample x 2 -> MiddleBlock -> Upsample x 2 -> Output conv
    Skip connections between downsample and upsample blocks.
    """

    def __init__(self, image_channels: int=1, n_channels: int=64, ch_mults: tuple=(1, 2, 4),
                 time_channels: int=64*4):
        super().__init__()

        # calculate channel sizes for each block
        c1 = n_channels * ch_mults[0] # 64
        c2 = n_channels * ch_mults[1] # 128
        c3 = n_channels * ch_mults[2] # 256

        # Time embedding module
        self.time_emb = TimeEmbedding(time_channels)

        # Input projection: 1ch -> 64ch
        self.image_proj = nn.Conv2d(image_channels, c1, kernel_size=3, padding=1)

        # Downsample blocks
        self.down1 = ResidualBlock(c1, c1, time_channels)
        self.downsample1 = Downsample(c1)
        self.down2 = ResidualBlock(c1, c2, time_channels)
        self.downsample2 = Downsample(c2)
        self.down3 = ResidualBlock(c2, c3, time_channels)

        # Middle block
        self.middle = MiddleBlock(c3, time_channels)

        # Upsample blocks - in_channels가 2배(concatenation 때문)인 것에 주의
        self.up3 = ResidualBlock(c3 + c3, c2, time_channels)
        self.upsample3 = Upsample(c2)
        self.up2 = ResidualBlock(c2 + c2, c1, time_channels)
        self.upsample2 = Upsample(c1)
        self.up1 = ResidualBlock(c1 + c1, c1, time_channels)

        # Final
        self.norm = nn.GroupNorm(8, c1)
        self.act = nn.SiLU()
        self.final = nn.Conv2d(c1, image_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, 28, 28) - noisy image
        t: (B,) - timesteps
        return: (B, 1, 28, 28) - predicted noise
        """

        # 1. Time embedding
        t = self.time_emb(t)

        # 2. Input projection
        x = self.image_proj(x)

        # 3. Downsample path
        h1 = self.down1(x,t)
        x = self.downsample1(h1)
        h2 = self.down2(x,t)
        x = self.downsample2(h2)
        h3 = self.down3(x, t)

        # 4. Middle block
        x = self.middle(h3, t)

        # 5. Upsample path (with skip connections)
        x = torch.cat((x, h3), dim=1) # concat along channel dimension
        x = self.up3(x, t)
        x = self.upsample3(x)

        x = torch.cat((x, h2), dim=1)
        x = self.up2(x, t)
        x = self.upsample2(x)

        x = torch.cat([x, h1], dim=1)
        x = self.up1(x, t)

        # 6. Final conv
        x = self.norm(x)
        x = self.act(x)
        x = self.final(x)

        return x
    
if __name__ == "__main__":
    print("\n=== UNet sanity check ===")
    # Create model
    model = UNet(
        image_channels=1,
        n_channels=64,
        ch_mults=(1, 2, 4),
        time_channels=64 * 4,
    )

    # Dummy inputs
    B = 4
    x = torch.randn(B, 1, 28, 28)
    t = torch.randint(0, 1000, (B,))  # 0~999 사이 정수 timestep

    print(f"Input  x: {x.shape}")
    print(f"Input  t: {t.shape}, dtype={t.dtype}")

    # Forward
    with torch.no_grad():
        out = model(x, t)

    print(f"Output:   {out.shape}")

    # Checks
    assert out.shape == x.shape, f"Shape mismatch! expected {x.shape}, got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"
    print("Shape OK & no NaN")

    # Parameter count (for sanity, MNIST UNet 정도면 ~수백만)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")