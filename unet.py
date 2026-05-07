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
    

if __name__ == "__main__":
    # 기존 TimeEmbedding 테스트
    embedding = TimeEmbedding(n_channels=128)
    t = torch.tensor([0, 100, 500, 999])
    t_emb = embedding(t)
    print(f"TimeEmbedding output: {t_emb.shape}")  # (4, 128)

    # ResidualBlock 테스트
    block = ResidualBlock(in_channels=64, out_channels=128, time_emb_dim=128)
    x = torch.randn(4, 64, 28, 28)  # 4장의 28x28 이미지, 64채널
    out = block(x, t_emb)
    print(f"ResidualBlock output: {out.shape}")  # (4, 128, 28, 28) 이어야 함
