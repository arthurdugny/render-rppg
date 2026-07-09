"""CNN1D_rPPG — version autonome (sans dépendance semaine 3/physnet)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_SCALARS = 5


class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class CNN1D_rPPG(nn.Module):
    """
    Entrée  : (B, in_channels, T)  — 12 canaux pour la paume (4 ROI × 3 RGB)
    Sorties : rPPG (B, T), scalars (B, N_SCALARS)
    """

    def __init__(self, in_channels=12, n_scalars=N_SCALARS, dropout=0.3):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(in_channels)

        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
        )
        self.res1  = ResBlock1D(64, kernel_size=7)
        self.pool1 = nn.MaxPool1d(2)

        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
        )
        self.res2  = ResBlock1D(128, kernel_size=5)
        self.pool2 = nn.MaxPool1d(2)

        self.enc3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
        )
        self.res3  = ResBlock1D(128, kernel_size=3)
        self.pool3 = nn.MaxPool1d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.rppg_conv  = nn.Conv1d(64, 1, kernel_size=1)
        self.scalar_head = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(64, n_scalars),
        )

    def forward(self, x):
        B, C, T = x.shape
        x = self.input_norm(x)
        x = self.pool1(self.res1(self.enc1(x)))
        x = self.pool2(self.res2(self.enc2(x)))
        x = self.pool3(self.res3(self.enc3(x)))
        x = self.bottleneck(x)

        rppg_raw = self.rppg_conv(x).squeeze(1)
        rppg_raw = F.interpolate(rppg_raw.unsqueeze(1), size=T, mode="linear",
                                  align_corners=False).squeeze(1)
        mu  = rppg_raw.mean(dim=1, keepdim=True)
        std = rppg_raw.std(dim=1,  keepdim=True).clamp(min=1e-6)
        rPPG = (rppg_raw - mu) / std

        scalars = self.scalar_head(x.mean(dim=2))
        return rPPG, scalars
