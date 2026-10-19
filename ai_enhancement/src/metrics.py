"""
Objective Evaluation Metrics for Speech Enhancement:
- SI-SDR: Scale-Invariant Signal-to-Distortion Ratio (dB)
- SNR: Signal-to-Noise Ratio (dB) & Delta SNR Improvement
- STOI: Short-Time Objective Intelligibility (0.0 to 1.0)
- PESQ: Perceptual Evaluation of Speech Quality (1.0 to 4.5, with optional fallback)
"""

from typing import Dict, Optional, Union
import torch
import numpy as np

try:
    import pystoi
    PYSTOI_AVAILABLE = True
except ImportError:
    PYSTOI_AVAILABLE = False

try:
    import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False


def compute_si_sdr(
    estimate: torch.Tensor,
    reference: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR).
    estimate: (B, T) or (T,)
    reference: (B, T) or (T,)
    returns: (B,) or scalar Tensor of SI-SDR values in dB.
    """
    if estimate.dim() == 1:
        estimate = estimate.unsqueeze(0)
    if reference.dim() == 1:
        reference = reference.unsqueeze(0)

    # Zero-mean normalization
    estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)
    reference = reference - torch.mean(reference, dim=-1, keepdim=True)

    # Optimal scaling factor alpha = <estimate, reference> / ||reference||^2
    dot_product = torch.sum(estimate * reference, dim=-1, keepdim=True)
    ref_energy = torch.sum(reference ** 2, dim=-1, keepdim=True) + eps
    alpha = dot_product / ref_energy

    # Target and noise components
    s_target = alpha * reference
    e_noise = estimate - s_target

    # 10 * log10(||s_target||^2 / ||e_noise||^2)
    target_power = torch.sum(s_target ** 2, dim=-1) + eps
    noise_power = torch.sum(e_noise ** 2, dim=-1) + eps

    si_sdr_val = 10.0 * torch.log10(target_power / noise_power)
    return si_sdr_val


def compute_snr(
    signal: torch.Tensor,
    noise_or_error: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Standard Signal-to-Noise Ratio (SNR) in dB.
    signal: Reference clean signal (B, T)
    noise_or_error: (B, T) - e.g., signal - estimate
    returns: (B,) SNR values in dB.
    """
    if signal.dim() == 1:
        signal = signal.unsqueeze(0)
    if noise_or_error.dim() == 1:
        noise_or_error = noise_or_error.unsqueeze(0)

    sig_pwr = torch.sum(signal ** 2, dim=-1) + eps
    noise_pwr = torch.sum(noise_or_error ** 2, dim=-1) + eps
    return 10.0 * torch.log10(sig_pwr / noise_pwr)


def compute_snr_improvement(
    enhanced: torch.Tensor,
    noisy: torch.Tensor,
    clean: torch.Tensor
) -> torch.Tensor:
    """
    Computes Delta SNR: SNR_enhanced - SNR_noisy.
    """
    snr_enhanced = compute_snr(clean, clean - enhanced)
    snr_noisy = compute_snr(clean, clean - noisy)
    return snr_enhanced - snr_noisy


def compute_stoi(
    clean: Union[torch.Tensor, np.ndarray],
    enhanced: Union[torch.Tensor, np.ndarray],
    sr: int = 16000
) -> Optional[float]:
    """
    Computes Short-Time Objective Intelligibility (STOI).
    clean: 1D waveform
    enhanced: 1D waveform
    returns: float score in [0.0, 1.0] (or None if pystoi not installed)
    """
    if not PYSTOI_AVAILABLE:
        return None

    if isinstance(clean, torch.Tensor):
        clean = clean.squeeze().detach().cpu().numpy()
    if isinstance(enhanced, torch.Tensor):
        enhanced = enhanced.squeeze().detach().cpu().numpy()

    # Ensure 1D float64
    clean = np.asarray(clean, dtype=np.float64)
    enhanced = np.asarray(enhanced, dtype=np.float64)

    min_len = min(len(clean), len(enhanced))
    clean = clean[:min_len]
    enhanced = enhanced[:min_len]

    try:
        score = pystoi.stoi(clean, enhanced, sr, extended=False)
        # Clip to valid [0.0, 1.0] range to prevent small negative float precision artifacts on pure tones
        clipped = float(np.clip(score, 0.0, 1.0))
        return clipped
    except Exception:
        return None


def compute_pesq(
    clean: Union[torch.Tensor, np.ndarray],
    enhanced: Union[torch.Tensor, np.ndarray],
    sr: int = 16000,
    mode: str = "wb"
) -> Optional[float]:
    """
    Computes Perceptual Evaluation of Speech Quality (PESQ).
    mode: 'wb' (wideband for 16kHz) or 'nb' (narrowband for 8kHz)
    returns: float score in [1.0, 4.5] (or None if pesq not installed)
    """
    if not PESQ_AVAILABLE:
        return None

    if isinstance(clean, torch.Tensor):
        clean = clean.squeeze().detach().cpu().numpy()
    if isinstance(enhanced, torch.Tensor):
        enhanced = enhanced.squeeze().detach().cpu().numpy()

    min_len = min(len(clean), len(enhanced))
    clean = clean[:min_len]
    enhanced = enhanced[:min_len]

    try:
        score = pesq.pesq(sr, clean, enhanced, mode)
        return float(score)
    except Exception:
        return None


def evaluate_batch(
    enhanced: torch.Tensor,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    sr: int = 16000
) -> Dict[str, float]:
    """
    Evaluates a full batch and returns averaged summary metrics.
    """
    with torch.no_grad():
        si_sdr_vals = compute_si_sdr(enhanced, clean)
        snr_imp_vals = compute_snr_improvement(enhanced, noisy, clean)

        metrics = {
            "si_sdr_db": float(torch.mean(si_sdr_vals).item()),
            "snr_improvement_db": float(torch.mean(snr_imp_vals).item())
        }

        # STOI evaluation
        if PYSTOI_AVAILABLE:
            stoi_list = []
            for b in range(clean.shape[0]):
                s_val = compute_stoi(clean[b], enhanced[b], sr=sr)
                if s_val is not None:
                    stoi_list.append(s_val)
            if stoi_list:
                metrics["stoi"] = float(np.mean(stoi_list))

        # PESQ evaluation
        if PESQ_AVAILABLE:
            pesq_list = []
            for b in range(clean.shape[0]):
                p_val = compute_pesq(clean[b], enhanced[b], sr=sr)
                if p_val is not None:
                    pesq_list.append(p_val)
            if pesq_list:
                metrics["pesq"] = float(np.mean(pesq_list))

        return metrics
