"""
Impulse-Aware and Multi-Resolution Loss Functions for Military Speech Enhancement.

Key Components:
1. ImpulseWeightedL1Loss:
   Penalizes reconstruction error with higher weights during acoustic impulse events
   (gunshots, artillery, explosions) detected via energy derivative of the noisy input.
   w(t) = 1.0 + alpha * ReLU(|x(t)| - |x(t-1)| - tau)

2. MultiResolutionSTFTLoss:
   Spectral convergence and log STFT magnitude loss over multiple window lengths
   to preserve speech harmonics and timbre.

3. CompositeDefenceLoss:
   Combines ImpulseWeightedL1Loss with MultiResolutionSTFTLoss.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_transient_weights(
    noisy_waveform: torch.Tensor,
    alpha: float = 5.0,
    tau: float = 0.1
) -> torch.Tensor:
    """
    Computes sample-wise transient weights based on the first-order energy derivative.
    noisy_waveform: (B, T) or (B, 1, T)
    alpha: Scaling factor for impulse penalty (e.g. 5.0 -> up to 5x higher weight)
    tau: Minimum threshold for energy derivative to trigger impulse penalty
    Returns:
        weights: (B, T) with baseline 1.0 and spikes >= 1.0 during impulses.
    """
    if noisy_waveform.dim() == 3:
        noisy_waveform = noisy_waveform.squeeze(1)

    abs_wav = torch.abs(noisy_waveform)
    # First-order difference: |x(t)| - |x(t-1)|
    diff = abs_wav[:, 1:] - abs_wav[:, :-1]
    # Pad left by 1 to maintain same length T
    diff = F.pad(diff, (1, 0), mode="constant", value=0.0)

    # w(t) = 1.0 + alpha * ReLU(diff - tau)
    weights = 1.0 + alpha * F.relu(diff - tau)
    return weights


class ImpulseWeightedL1Loss(nn.Module):
    """
    Impulse-Weighted L1 Waveform Loss:
    L = (1 / T) * sum_{t=1}^T w(t) * | s_hat(t) - s_clean(t) |
    """
    def __init__(self, alpha: float = 5.0, tau: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.tau = tau

    def forward(
        self,
        pred_waveform: torch.Tensor,
        clean_waveform: torch.Tensor,
        noisy_waveform: torch.Tensor
    ) -> torch.Tensor:
        """
        pred_waveform: (B, T)
        clean_waveform: (B, T)
        noisy_waveform: (B, T)
        """
        if pred_waveform.dim() == 3:
            pred_waveform = pred_waveform.squeeze(1)
        if clean_waveform.dim() == 3:
            clean_waveform = clean_waveform.squeeze(1)
        if noisy_waveform.dim() == 3:
            noisy_waveform = noisy_waveform.squeeze(1)

        weights = compute_transient_weights(noisy_waveform, alpha=self.alpha, tau=self.tau)
        abs_diff = torch.abs(pred_waveform - clean_waveform)
        weighted_loss = torch.mean(weights * abs_diff)
        return weighted_loss


class STFTLoss(nn.Module):
    """
    Single-resolution STFT loss computing spectral convergence and log magnitude loss.
    """
    def __init__(self, fft_size: int = 512, hop_size: int = 128, win_size: int = 512):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.register_buffer("window", torch.hann_window(win_size))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)

        pred_stft = torch.stft(
            pred,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window,
            center=True,
            return_complex=True
        )
        target_stft = torch.stft(
            target,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window,
            center=True,
            return_complex=True
        )

        pred_mag = torch.abs(pred_stft) + 1e-7
        target_mag = torch.abs(target_stft) + 1e-7

        # Spectral convergence: Frobenius norm difference divided by target norm
        sc_loss = torch.norm(target_mag - pred_mag, p="fro") / (torch.norm(target_mag, p="fro") + 1e-7)

        # Log magnitude loss: L1 distance between log-magnitudes
        mag_loss = F.l1_loss(torch.log(pred_mag), torch.log(target_mag))

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT Loss computed across multiple window resolutions.
    """
    def __init__(
        self,
        fft_sizes: List[int] = [320, 640, 1280],
        hop_sizes: List[int] = [80, 160, 320],
        win_sizes: List[int] = [320, 640, 1280],
    ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_sizes)
        self.losses = nn.ModuleList([
            STFTLoss(fft, hop, win)
            for fft, hop, win in zip(fft_sizes, hop_sizes, win_sizes)
        ])

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for loss_fn in self.losses:
            sc_loss, mag_loss = loss_fn(pred, target)
            total_loss += (sc_loss + mag_loss)
        return total_loss / len(self.losses)


class CompositeDefenceLoss(nn.Module):
    """
    Composite Loss for Military Speech Enhancement:
    L = L_impulse_l1 + lambda_mrstft * L_mrstft
    """
    def __init__(
        self,
        alpha: float = 5.0,
        tau: float = 0.1,
        lambda_mrstft: float = 0.5
    ):
        super().__init__()
        self.impulse_loss = ImpulseWeightedL1Loss(alpha=alpha, tau=tau)
        self.mrstft_loss = MultiResolutionSTFTLoss()
        self.lambda_mrstft = lambda_mrstft

    def forward(
        self,
        pred_waveform: torch.Tensor,
        clean_waveform: torch.Tensor,
        noisy_waveform: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        l_impulse = self.impulse_loss(pred_waveform, clean_waveform, noisy_waveform)
        l_mrstft = self.mrstft_loss(pred_waveform, clean_waveform)
        total_loss = l_impulse + self.lambda_mrstft * l_mrstft

        loss_dict = {
            "total_loss": total_loss.item(),
            "impulse_loss": l_impulse.item(),
            "mrstft_loss": l_mrstft.item()
        }
        return total_loss, loss_dict
