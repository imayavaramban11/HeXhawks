"""
Unit Tests for Filtered-x Normalized Least Mean Squares (FxNLMS) and Parallel Hybrid System.
"""

import pytest
import torch
import numpy as np

from src.config import Config
from src.model import CausalFullSubNetPlus
from src.dsp_fxnlms import FxNLMSFilter, ParallelHybridSystem

def test_fxnlms_initialization_and_reset():
    filter_mod = FxNLMSFilter(filter_len=32, secondary_filter_len=16)
    assert filter_mod.weights.shape == (32,)
    assert torch.all(filter_mod.weights == 0.0)

    # Process a few samples (greater than secondary path delay of 4 samples)
    for _ in range(10):
        filter_mod.step(1.0, 0.5)
    assert not torch.all(filter_mod.weights == 0.0)

    # Reset
    filter_mod.reset()
    assert torch.all(filter_mod.weights == 0.0)
    assert torch.all(filter_mod.x_buf == 0.0)

def test_fxnlms_sinusoidal_convergence():
    """
    Verifies that FxNLMS adapts to low-frequency engine hum and achieves > 15 dB attenuation.
    """
    sr = 16000
    T_samples = 16000  # 1 second
    t = torch.linspace(0, 1.0, T_samples)

    # Reference noise: low-frequency engine rumble (120Hz + 240Hz harmonics)
    x = torch.sin(2 * torch.pi * 120.0 * t) + 0.5 * torch.sin(2 * torch.pi * 240.0 * t)

    # Secondary path model (delay of 4 samples with attenuation 0.85)
    sec_path = torch.zeros(32)
    sec_path[4] = 0.85

    # Primary path: leaked noise arrives at error mic with acoustic delay of 6 samples
    d = torch.roll(x, 6) * 0.7

    anc = FxNLMSFilter(filter_len=64, secondary_filter_len=32, mu=0.08)
    anc.set_secondary_path(sec_path)

    err, anti_noise = anc.filter(x, d)

    # Verify attenuation over the final 4000 samples (after filter has converged)
    initial_pwr = torch.mean(d[:1000] ** 2).item()
    converged_pwr = torch.mean(err[0, -2000:] ** 2).item()

    attenuation_db = 10.0 * np.log10(initial_pwr / (converged_pwr + 1e-8))
    print(f"FxNLMS Attenuation achieved: {attenuation_db:.2f} dB")

    # Should achieve substantial cancellation on stationary low-frequency rumble
    assert attenuation_db > 15.0, f"Expected > 15 dB attenuation, got {attenuation_db:.2f} dB"

def test_parallel_hybrid_system_forward():
    """
    Verifies coordinated execution of Path 1 (AI) and Path 2 (FxNLMS).
    """
    config = Config()
    ai_model = CausalFullSubNetPlus(config)
    hybrid_system = ParallelHybridSystem(ai_model=ai_model)

    B = 2
    T = 16000  # 1 second of audio
    primary_mic = torch.randn(B, T)
    earcup_leak = torch.randn(B, T) * 0.3

    telemetry = hybrid_system(primary_mic, earcup_leak)

    assert "enhanced_speech" in telemetry
    assert "anti_noise" in telemetry
    assert "residual_noise" in telemetry
    assert "soldier_perceived_audio" in telemetry
    assert "anc_attenuation_db" in telemetry

    # Check shapes
    assert telemetry["enhanced_speech"].shape == (B, T)
    assert telemetry["anti_noise"].shape == (B, T)
    assert telemetry["residual_noise"].shape == (B, T)
    assert telemetry["soldier_perceived_audio"].shape == (B, T)
    assert isinstance(telemetry["anc_attenuation_db"], float)
