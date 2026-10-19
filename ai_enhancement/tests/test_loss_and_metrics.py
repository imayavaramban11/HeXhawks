"""
Unit Tests for Impulse-Aware Loss Functions and Evaluation Metrics.
"""

import pytest
import torch
import numpy as np

from src.loss import (
    compute_transient_weights,
    ImpulseWeightedL1Loss,
    MultiResolutionSTFTLoss,
    CompositeDefenceLoss
)
from src.metrics import (
    compute_si_sdr,
    compute_snr,
    compute_snr_improvement,
    compute_stoi,
    evaluate_batch,
    PYSTOI_AVAILABLE
)

def test_transient_weights_detection():
    # 1 second of audio at 16kHz
    T = 16000
    noisy = torch.zeros(1, T)
    # Inject gunshot/impulse at sample 8000
    noisy[0, 8000] = 0.95

    weights = compute_transient_weights(noisy, alpha=5.0, tau=0.1)

    # Steady regions should have weight 1.0
    assert torch.all(weights[0, :7999] == 1.0)
    assert torch.all(weights[0, 8002:] == 1.0)

    # The impulse at 8000 should have amplified weight
    assert weights[0, 8000].item() > 1.0
    # Expected: 1.0 + 5.0 * (0.95 - 0.1) = 1.0 + 4.25 = 5.25
    assert torch.isclose(weights[0, 8000], torch.tensor(5.25))

def test_impulse_weighted_l1_loss_gradient():
    loss_fn = ImpulseWeightedL1Loss(alpha=5.0, tau=0.1)

    clean = torch.randn(2, 8000)
    noisy = clean + torch.randn(2, 8000) * 0.2
    # Inject impulse into noisy
    noisy[:, 4000] = 1.0

    pred = clean.clone().detach().requires_grad_(True)

    loss = loss_fn(pred, clean, noisy)
    loss.backward()

    assert pred.grad is not None
    assert not torch.isnan(loss)

def test_multi_resolution_stft_loss():
    loss_fn = MultiResolutionSTFTLoss()

    clean = torch.randn(2, 16000)
    # Identical signals should yield near-zero loss
    zero_loss = loss_fn(clean, clean)
    assert zero_loss.item() < 1e-4

    # Degraded signal should yield positive loss
    degraded = clean + torch.randn(2, 16000) * 0.5
    deg_loss = loss_fn(degraded, clean)
    assert deg_loss.item() > 0.1

def test_composite_defence_loss():
    comp_loss = CompositeDefenceLoss(alpha=5.0, tau=0.1, lambda_mrstft=0.5)

    clean = torch.randn(2, 8000)
    noisy = clean + torch.randn(2, 8000) * 0.3
    pred = clean + torch.randn(2, 8000) * 0.1

    total, metrics = comp_loss(pred, clean, noisy)
    assert "total_loss" in metrics
    assert "impulse_loss" in metrics
    assert "mrstft_loss" in metrics
    assert total.item() > 0.0

def test_si_sdr_scale_invariance():
    """
    Mathematical proof of Scale-Invariance:
    SI-SDR(c * s, s) must be identical for any positive scalar c.
    """
    clean = torch.randn(2, 16000)
    est_1 = clean + torch.randn(2, 16000) * 0.05
    est_scaled = est_1 * 4.2  # arbitrary volume scaling

    sdr_1 = compute_si_sdr(est_1, clean)
    sdr_scaled = compute_si_sdr(est_scaled, clean)

    assert torch.allclose(sdr_1, sdr_scaled, atol=1e-3)

def test_snr_improvement():
    clean = torch.randn(2, 16000)
    noisy = clean + torch.randn(2, 16000) * 1.0  # high noise
    enhanced = clean + torch.randn(2, 16000) * 0.1  # low noise

    delta_snr = compute_snr_improvement(enhanced, noisy, clean)
    # Enhanced should have positive improvement over noisy
    assert torch.all(delta_snr > 10.0)

def test_stoi_and_batch_evaluation():
    sr = 16000
    t = np.linspace(0, 1, sr)
    clean = np.sin(2 * np.pi * 440 * t)
    noisy = clean + np.random.randn(sr) * 0.3
    clean_t = torch.from_numpy(clean).unsqueeze(0).float()
    noisy_t = torch.from_numpy(noisy).unsqueeze(0).float()
    enhanced_t = clean_t + torch.randn_like(clean_t) * 0.05

    metrics = evaluate_batch(enhanced_t, clean_t, noisy_t, sr=sr)
    assert "si_sdr_db" in metrics
    assert "snr_improvement_db" in metrics

    if PYSTOI_AVAILABLE:
        assert "stoi" in metrics
        assert 0.0 <= metrics["stoi"] <= 1.0
