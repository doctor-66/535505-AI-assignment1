import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class TRFNet(nn.Module):
    """
    Input:
        x: (B, 30, T)

    Output:
        trf_global:   (30, 500)
        trf_residual: (B, 30, 500)
        trf_final:    (B, 30, 500)
    """
    def __init__(
        self,
        in_channels=30,
        trf_len=500,
        hidden_dim=64,
        latent_dim=128,
        context_dim=0,
        dropout=0.1,
        residual_scale=0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.trf_len = trf_len
        self.context_dim = context_dim

        # -------- Global linear TRF --------
        # shape: (30, 500)
        self.global_trf = nn.Parameter(torch.zeros(in_channels, trf_len))

        # -------- Nonlinear encoder --------
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            ConvBlock1D(hidden_dim, hidden_dim, kernel_size=3, dilation=1, dropout=dropout),
            ConvBlock1D(hidden_dim, hidden_dim, kernel_size=3, dilation=2, dropout=dropout),
            ConvBlock1D(hidden_dim, hidden_dim, kernel_size=3, dilation=4, dropout=dropout),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        # latent projector
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, latent_dim),
            nn.Tanh(),
            # nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
            # nn.ReLU(inplace=True),
        )

        # residual TRF generator
        self.residual_head = nn.Linear(latent_dim, in_channels * trf_len)

        # learnable scalar gate for residual branch
        self.residual_gate = nn.Parameter(torch.tensor(residual_scale, dtype=torch.float32))

    def forward(self, x, context=None):
        B = x.shape[0]

        feat = self.encoder(x)                     # (B, hidden_dim, T)
        pooled = self.pool(feat).squeeze(-1)      # (B, hidden_dim)
        
        if self.context_dim > 0:
            if context is None:
                raise ValueError("context is required when context_dim > 0")

            context = context.to(x.device).float()

            # 若是 (B,) -> (B,1)
            if context.ndim == 1:
                context = context.unsqueeze(1)

            pooled = torch.cat([pooled, context], dim=1)

        z = self.to_latent(pooled)

        trf_residual = self.residual_head(z)
        trf_residual = trf_residual.view(B, self.in_channels, self.trf_len)

        trf_global_expand = self.global_trf.unsqueeze(0).expand(B, -1, -1)
        trf_final = trf_global_expand + self.residual_gate * trf_residual

        return trf_global_expand, trf_residual, trf_final