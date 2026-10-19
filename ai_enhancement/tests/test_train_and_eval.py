"""
Unit Tests for Training and Evaluation Pipelines.
"""

import os
import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW

from src.config import Config
from src.model import CausalFullSubNetPlus
from src.loss import CompositeDefenceLoss
from src.dataset import create_dataloaders, DefenceSpeechDataset
from train import train_one_epoch, validate, save_checkpoint
from evaluate import run_benchmark

def test_train_one_epoch_step():
    config = Config()
    config.data.segment_length = 8000
    config.data.batch_size = 2
    device = torch.device("cpu")

    model = CausalFullSubNetPlus(config).to(device)
    criterion = CompositeDefenceLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    train_loader, _ = create_dataloaders(config=config, virtual_size=4)

    initial_param = model.subband_lstm.weight_ih_l0.clone()

    res = train_one_epoch(
        model=model,
        loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        max_grad_norm=5.0
    )

    assert "train_loss" in res
    assert res["train_loss"] > 0.0

    # Ensure weights updated
    assert not torch.equal(initial_param, model.subband_lstm.weight_ih_l0)

def test_validation_loop():
    config = Config()
    config.data.segment_length = 8000
    config.data.batch_size = 2
    device = torch.device("cpu")

    model = CausalFullSubNetPlus(config).to(device)
    criterion = CompositeDefenceLoss().to(device)
    _, val_loader = create_dataloaders(config=config, virtual_size=4)

    res = validate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        sr=config.audio.sample_rate
    )

    assert "val_loss" in res
    assert "val_si_sdr_db" in res
    assert "val_snr_imp_db" in res

def test_checkpoint_save_and_load(tmp_path):
    config = Config()
    device = torch.device("cpu")
    model = CausalFullSubNetPlus(config).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)

    chk_path = os.path.join(tmp_path, "test_checkpoint.pt")
    save_checkpoint(chk_path, model, optimizer, epoch=3, best_si_sdr=12.5, config=config)
    assert os.path.exists(chk_path)

    # Load back
    loaded_chk = torch.load(chk_path, map_location=device, weights_only=False)
    assert loaded_chk["epoch"] == 3
    assert loaded_chk["best_si_sdr"] == 12.5

    model_new = CausalFullSubNetPlus(config).to(device)
    model_new.load_state_dict(loaded_chk["model_state_dict"])
    assert torch.equal(model.subband_lstm.weight_ih_l0, model_new.subband_lstm.weight_ih_l0)

def test_run_benchmark_integration():
    config = Config()
    config.data.segment_length = 8000
    device = torch.device("cpu")
    model = CausalFullSubNetPlus(config).to(device)
    dataset = DefenceSpeechDataset(config=config.data, virtual_size=2)

    results = run_benchmark(model=model, dataset=dataset, device=device, num_samples=2, sr=config.audio.sample_rate)

    assert "unprocessed_si_sdr_db" in results
    assert "enhanced_si_sdr_db" in results
    assert "anc_low_freq_attenuation_db" in results
    assert results["samples_evaluated"] == 2
