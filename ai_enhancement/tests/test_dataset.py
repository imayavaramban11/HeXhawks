"""
Unit Tests for Dynamic Data Pipeline, SNR Mixer, and Audio Dataset.
"""

import pytest
import torch
from src.config import Config, DataConfig
from src.dataset import (
    mix_audio_at_snr,
    inject_impulsive_noise,
    generate_synthetic_speech,
    generate_synthetic_defence_noise,
    DefenceSpeechDataset,
    create_dataloaders
)

def test_mix_audio_at_snr_precision():
    clean = torch.randn(1, 48000)
    noise = torch.randn(1, 48000)

    for target_snr in [-5.0, 0.0, 5.0, 15.0]:
        noisy, clean_out, actual_snr = mix_audio_at_snr(clean, noise, snr_db=target_snr)
        assert abs(actual_snr - target_snr) < 0.1, f"SNR mismatch! Target: {target_snr}, Actual: {actual_snr}"
        assert noisy.shape == clean_out.shape == (1, 48000)
        # Ensure no clipping beyond [-1.0, 1.0]
        assert torch.max(torch.abs(noisy)).item() <= 1.0

def test_inject_impulsive_noise():
    noisy = torch.zeros(1, 48000)
    impulse = torch.ones(1, 1600) * 0.8
    offset = 10000

    blasted, actual_offset = inject_impulsive_noise(noisy, impulse, offset=offset, gain_multiplier=2.0)
    assert blasted.shape == (1, 48000)
    assert actual_offset == offset
    # Before offset should be 0.0
    assert torch.all(blasted[:, :offset] == 0.0)
    # At offset should have energy
    assert torch.any(blasted[:, offset:offset + 1600] > 0.0)

def test_synthetic_generators():
    speech = generate_synthetic_speech(duration_sec=1.0, sr=16000)
    assert speech.shape == (1, 16000)
    assert torch.max(torch.abs(speech)).item() > 0.0

    engine = generate_synthetic_defence_noise(duration_sec=1.0, sr=16000, noise_type="engine")
    assert engine.shape == (1, 16000)

    rotor = generate_synthetic_defence_noise(duration_sec=1.0, sr=16000, noise_type="rotor")
    assert rotor.shape == (1, 16000)

    gunshot = generate_synthetic_defence_noise(duration_sec=0.2, sr=16000, noise_type="gunshot")
    assert gunshot.shape == (1, 3200)

def test_defence_speech_dataset():
    cfg = DataConfig(segment_length=16000, batch_size=2)
    dataset = DefenceSpeechDataset(config=cfg, virtual_size=10)

    assert len(dataset) == 10
    sample = dataset[0]

    assert "noisy" in sample
    assert "clean" in sample
    assert "snr_db" in sample
    assert "has_impulse" in sample

    assert sample["noisy"].shape == (16000,)
    assert sample["clean"].shape == (16000,)
    assert isinstance(sample["snr_db"], float)
    assert isinstance(sample["has_impulse"], bool)

def test_create_dataloaders():
    cfg = Config()
    cfg.data.segment_length = 16000
    cfg.data.batch_size = 2

    train_loader, val_loader = create_dataloaders(config=cfg, virtual_size=8, train_split=0.75)

    assert len(train_loader) == 3  # 6 samples / batch 2
    assert len(val_loader) == 1    # 2 samples / batch 2

    batch = next(iter(train_loader))
    assert batch["noisy"].shape == (2, 16000)
    assert batch["clean"].shape == (2, 16000)
