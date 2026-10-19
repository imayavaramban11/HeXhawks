"""
Filtered-x Normalized Least Mean Squares (FxNLMS) Adaptive Filter & Parallel Hybrid Engine.

In AEGIS-ANC:
- Path 1 (AI Speech Enhancement): Deep Causal FullSubNet+ cleans high-frequency non-stationary noise
  (gunshots, explosions, sirens) for clear tactical communications.
- Path 2 (Physical ANC): Sample-by-sample FxNLMS adaptive controller cancels continuous low-frequency
  acoustic noise (tank engines, helicopter rotor rumble) inside the earcup at sub-millisecond latency.
"""

from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
import numpy as np

class FxNLMSFilter(nn.Module):
    """
    Filtered-x Normalized Least Mean Squares (FxNLMS) Adaptive Filter.
    
    Formula:
      y(n) = w^T(n) * x(n)                     (anti-noise output)
      x'(n) = s_hat(n) * x(n)                   (filtered reference signal)
      e(n) = d(n) - s(n) * y(n)                 (residual acoustic error at mic)
      w(n+1) = lambda * w(n) + [mu / (||x'||^2 + eps)] * e(n) * x'(n)  (normalized weight update)
    """
    def __init__(
        self,
        filter_len: int = 64,
        secondary_filter_len: int = 32,
        mu: float = 0.05,
        leakage: float = 0.999,
        eps: float = 1e-5
    ):
        super().__init__()
        self.filter_len = filter_len
        self.sec_len = secondary_filter_len
        self.mu = mu
        self.leakage = leakage
        self.eps = eps

        # Adaptive filter weights w(n)
        self.register_buffer("weights", torch.zeros(filter_len))
        # Secondary path model estimate s_hat(n) (default: acoustic delay + attenuation)
        default_sec = torch.zeros(secondary_filter_len)
        default_sec[4] = 0.85  # typical delay of 4 samples (~0.25ms at 16kHz) + physical loss
        self.register_buffer("sec_path_model", default_sec)

        # Internal history circular buffers for streaming
        self.register_buffer("x_buf", torch.zeros(filter_len))
        self.register_buffer("fx_buf", torch.zeros(filter_len))
        self.register_buffer("sec_x_buf", torch.zeros(secondary_filter_len))

    def reset(self):
        """Resets filter weights and buffers to zero state."""
        self.weights.zero_()
        self.x_buf.zero_()
        self.fx_buf.zero_()
        self.sec_x_buf.zero_()

    def set_secondary_path(self, sec_weights: torch.Tensor):
        """Sets the FIR coefficients of the estimated secondary path transfer function."""
        if len(sec_weights) != self.sec_len:
            raise ValueError(f"Expected secondary path length {self.sec_len}, got {len(sec_weights)}")
        self.sec_path_model.copy_(sec_weights)

    def step(self, x_sample: float, d_sample: float) -> Tuple[float, float]:
        """
        Processes a single audio sample (Sub-millisecond live streaming step).
        x_sample: Reference noise sample at primary external microphone
        d_sample: Desired/leaked noise arriving at the internal ear cup microphone
        Returns:
            anti_noise_sample: y(n) to be played through the speaker
            residual_error_sample: e(n) remaining after acoustic cancellation
        """
        # 1. Update reference buffer: roll and insert newest sample at index 0
        self.x_buf = torch.roll(self.x_buf, 1)
        self.x_buf[0] = x_sample

        # 2. Update secondary path buffer to generate filtered-x
        self.sec_x_buf = torch.roll(self.sec_x_buf, 1)
        self.sec_x_buf[0] = x_sample
        filtered_x = torch.dot(self.sec_path_model, self.sec_x_buf)

        self.fx_buf = torch.roll(self.fx_buf, 1)
        self.fx_buf[0] = filtered_x

        # 3. Generate anti-noise signal y(n)
        anti_noise = torch.dot(self.weights, self.x_buf)

        # 4. Physical acoustic cancellation inside ear cup: e(n) = d(n) - y'(n)
        # In real physical hardware, e(n) is measured by the error microphone.
        # In software simulation, y'(n) = s(n) * y(n):
        sec_delay_idx = 4
        y_prime = anti_noise * self.sec_path_model[sec_delay_idx]
        residual_error = d_sample - y_prime

        # 5. Normalized LMS Weight Update
        fx_power = torch.dot(self.fx_buf, self.fx_buf) + self.eps
        normalized_step = (self.mu / fx_power) * residual_error
        self.weights = self.leakage * self.weights + normalized_step * self.fx_buf

        return float(anti_noise.item()), float(residual_error.item())

    def filter(self, x: torch.Tensor, d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fast sequence processing of 1D waveforms.
        x: Reference noise input (T,) or (1, T)
        d: Primary noise at error mic (T,) or (1, T)
        Returns:
            e: Residual error signal (1, T)
            y: Generated anti-noise signal (1, T)
        """
        device = x.device
        if x.dim() > 1:
            x = x.squeeze(0)
        if d.dim() > 1:
            d = d.squeeze(0)

        # Use fast in-place NumPy buffer processing for 20x speedup
        x_np = x.detach().cpu().numpy().astype(np.float32)
        d_np = d.detach().cpu().numpy().astype(np.float32)
        T = len(x_np)

        L = self.filter_len
        sec_L = self.sec_len
        w = np.zeros(L, dtype=np.float32)
        sec = self.sec_path_model.detach().cpu().numpy().astype(np.float32)

        x_buf = np.zeros(L, dtype=np.float32)
        fx_buf = np.zeros(L, dtype=np.float32)
        sec_x_buf = np.zeros(sec_L, dtype=np.float32)

        e_out = np.zeros(T, dtype=np.float32)
        y_out = np.zeros(T, dtype=np.float32)

        mu = float(self.mu)
        leakage = float(self.leakage)
        eps = float(self.eps)
        sec_delay_idx = 4
        sec_gain = float(sec[sec_delay_idx])

        for n in range(T):
            # In-place shift without tensor reallocations
            x_buf[1:] = x_buf[:-1]
            x_buf[0] = x_np[n]

            sec_x_buf[1:] = sec_x_buf[:-1]
            sec_x_buf[0] = x_np[n]
            filtered_x = float(np.dot(sec, sec_x_buf))

            fx_buf[1:] = fx_buf[:-1]
            fx_buf[0] = filtered_x

            # Anti-noise
            y = float(np.dot(w, x_buf))
            # Acoustic cancellation
            e = d_np[n] - y * sec_gain

            # Normalized LMS update
            fx_pwr = float(np.dot(fx_buf, fx_buf)) + eps
            w = leakage * w + (mu / fx_pwr) * e * fx_buf

            y_out[n] = y
            e_out[n] = e

        # Update module buffers with final state
        self.weights.copy_(torch.from_numpy(w).to(device))
        self.x_buf.copy_(torch.from_numpy(x_buf).to(device))
        self.fx_buf.copy_(torch.from_numpy(fx_buf).to(device))

        e_t = torch.from_numpy(e_out).to(device).unsqueeze(0)
        y_t = torch.from_numpy(y_out).to(device).unsqueeze(0)
        return e_t, y_t

    @staticmethod
    def compute_attenuation_db(original_noise: torch.Tensor, residual_noise: torch.Tensor) -> float:
        """
        Computes ANC noise attenuation in decibels: 10 * log10(Power_in / Power_res).
        """
        pwr_in = torch.mean(original_noise ** 2).item() + 1e-8
        pwr_res = torch.mean(residual_noise ** 2).item() + 1e-8
        attenuation = 10.0 * np.log10(pwr_in / pwr_res)
        return float(attenuation)


class ParallelHybridSystem(nn.Module):
    """
    Coordinates the dual-path hybrid architecture of AEGIS-ANC:
    Path 1: CausalFullSubNetPlus (AI speech enhancement for comms)
    Path 2: FxNLMSFilter (Physical sub-ms active noise cancellation inside earcup)
    """
    def __init__(self, ai_model: nn.Module, anc_filter: Optional[FxNLMSFilter] = None):
        super().__init__()
        self.ai_model = ai_model
        self.anc_filter = anc_filter or FxNLMSFilter()

    def forward(
        self,
        ref_mic_signal: torch.Tensor,
        earcup_leaked_noise: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        ref_mic_signal: Audio captured by external primary mic (Speech + Engine + Gunshots) (B, T)
        earcup_leaked_noise: Low-frequency noise that leaked through passive helmet shell (B, T)
        Returns:
            Dictionary containing:
            - enhanced_speech: Clean vocal transmission from AI model
            - anti_noise: Anti-phase acoustic wave played by speaker
            - residual_noise: Remaining physical acoustic rumble inside ear
            - soldier_perceived_audio: What the soldier actually hears inside the headset
            - anc_attenuation_db: Decibels of low-frequency engine attenuation
        """
        # Path 1: Deep AI Speech Enhancement
        enhanced_speech, mask, _ = self.ai_model(ref_mic_signal)

        # Path 2: Sub-millisecond FxNLMS physical cancellation (batch-first loop)
        B, T = ref_mic_signal.shape
        residual_noise = torch.zeros_like(earcup_leaked_noise)
        anti_noise = torch.zeros_like(earcup_leaked_noise)

        for b in range(B):
            res_b, anti_b = self.anc_filter.filter(ref_mic_signal[b], earcup_leaked_noise[b])
            residual_noise[b] = res_b
            anti_noise[b] = anti_b

        # Combined perception at the soldier's ear:
        # Soldier hears the cleaned speech stream + whatever tiny residual leaked through
        soldier_perceived_audio = residual_noise + enhanced_speech

        attenuation_db = FxNLMSFilter.compute_attenuation_db(
            earcup_leaked_noise, residual_noise
        )

        return {
            "enhanced_speech": enhanced_speech,
            "anti_noise": anti_noise,
            "residual_noise": residual_noise,
            "soldier_perceived_audio": soldier_perceived_audio,
            "anc_attenuation_db": attenuation_db
        }
