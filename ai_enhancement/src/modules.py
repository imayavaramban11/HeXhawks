"""
Core Neural Modules for Causal FullSubNet+ Architecture:
- CausalConv1d: Causal 1D convolution with zero future lookahead
- ChannelAttention1d: MulCA channel attention for acoustic feature recalibration
- CausalTCNBlock: Dilated causal residual block
- CausalFullBandExtractor: Extracts global acoustic context across full frequency band
- SubBandFeatureExtractor: Vectorized extraction of local frequency bands
- ComplexMasking: Complex Ideal Ratio Mask (cIRM) application
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    """
    1D Convolution with strict causal padding.
    Pads ONLY on the left (past context) to guarantee zero future lookahead.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1, bias: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.causal_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        returns: (B, C_out, T)
        """
        if self.causal_padding > 0:
            x = F.pad(x, (self.causal_padding, 0))
        return self.conv(x)


class ChannelAttention1d(nn.Module):
    """
    Multiplicative Channel Attention (MulCA).
    Recalibrates channel weights based on global time-averaged statistics.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced_dim = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_dim),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        returns: (B, C, T)
        """
        # Global average across time
        gap = torch.mean(x, dim=-1)  # (B, C)
        weights = self.fc(gap).unsqueeze(-1)  # (B, C, 1)
        return x * weights


class CausalTCNBlock(nn.Module):
    """
    Residual Dilated Causal Convolutional Block with Channel Attention.
    """
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, reduction: int = 4, dropout: float = 0.0):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.prelu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.conv2 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(channels)
        self.prelu2 = nn.PReLU()

        self.attention = ChannelAttention1d(channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        returns: (B, C, T)
        """
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.prelu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.prelu2(out)

        out = self.attention(out)
        return out + residual


class CausalFullBandExtractor(nn.Module):
    """
    Full-Band Feature Extractor.
    Takes input spectrogram features across all frequency bins (e.g., Mag, Real, Imag)
    and uses stacked causal dilated TCN blocks to produce a global acoustic embedding.
    """
    def __init__(self, in_features: int, hidden_dim: int = 64, num_layers: int = 3, reduction: int = 4, dropout: float = 0.0):
        super().__init__()
        self.input_proj = nn.Conv1d(in_features, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList([
            CausalTCNBlock(
                channels=hidden_dim,
                kernel_size=3,
                dilation=2**i,
                reduction=reduction,
                dropout=dropout
            )
            for i in range(num_layers)
        ])
        self.output_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, in_features, T) where in_features = 3 * F (Mag, Real, Imag)
        returns: (B, hidden_dim, T)
        """
        out = self.input_proj(x)
        for block in self.blocks:
            out = block(out)
        out = self.output_norm(out)
        return out


class SubBandFeatureExtractor(nn.Module):
    """
    Vectorized Sub-Band Extractor.
    For each frequency bin f, extracts a local window of (2*K + 1) bins
    across all input channels (Magnitude, Real, Imag).
    Zero-pads boundary frequency bins so output shape matches F.
    """
    def __init__(self, num_neighbors: int = 1):
        super().__init__()
        self.k = num_neighbors

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, F, T) - e.g., C=3 (Mag, Real, Imag)
        returns: (B, F, T, C * (2*K + 1))
        """
        B, C, F_dim, T = x.shape
        # Pad along the frequency dimension (dim=2): left and right by k
        padded = F.pad(x, (0, 0, self.k, self.k), mode="constant", value=0.0)
        # Unfold along frequency dimension: size = 2*k + 1, step = 1
        windows = padded.unfold(dimension=2, size=2 * self.k + 1, step=1)  # (B, C, F_dim, T, 2K+1)
        # Permute to (B, F_dim, T, C, 2K+1) and flatten the last two dimensions
        subband = windows.permute(0, 2, 3, 1, 4).contiguous().view(B, F_dim, T, C * (2 * self.k + 1))
        return subband


def apply_complex_mask(noisy_spec: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Applies Complex Ideal Ratio Mask (cIRM) to noisy complex STFT via complex multiplication:
    S_enhanced = (X_real + j*X_imag) * (M_real + j*M_imag)
               = (X_real*M_real - X_imag*M_imag) + j*(X_real*M_imag + X_imag*M_real)

    noisy_spec: Complex tensor of shape (B, F, T)
    mask: Float tensor of shape (B, F, T, 2) where mask[..., 0] is Real and mask[..., 1] is Imag
    returns: Enhanced complex tensor of shape (B, F, T)
    """
    x_r = noisy_spec.real
    x_i = noisy_spec.imag
    m_r = mask[..., 0]
    m_i = mask[..., 1]

    s_r = x_r * m_r - x_i * m_i
    s_i = x_r * m_i + x_i * m_r

    return torch.complex(s_r, s_i)
