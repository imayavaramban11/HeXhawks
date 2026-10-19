"""
Training Engine for AEGIS-ANC (Causal FullSubNet+ with Impulse-Aware Loss).

Usage:
    python train.py --epochs 10 --batch-size 4 --lr 5e-4
    python train.py --resume checkpoints/latest_checkpoint.pt
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import argparse
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.config import Config
from src.model import CausalFullSubNetPlus
from src.loss import CompositeDefenceLoss
from src.metrics import evaluate_batch
from src.dataset import create_dataloaders


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 5.0
) -> Dict[str, float]:
    """Runs one training epoch over the dataloader."""
    model.train()
    running_loss = 0.0
    running_impulse_loss = 0.0
    running_mrstft_loss = 0.0
    total_batches = len(loader)

    for batch_idx, batch in enumerate(loader):
        noisy = batch["noisy"].to(device)
        clean = batch["clean"].to(device)

        optimizer.zero_grad()
        enhanced, _, _ = model(noisy)

        loss, loss_dict = criterion(enhanced, clean, noisy)
        loss.backward()

        # Gradient clipping to stabilize training against extreme audio spikes
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()

        running_loss += loss_dict["total_loss"]
        running_impulse_loss += loss_dict["impulse_loss"]
        running_mrstft_loss += loss_dict["mrstft_loss"]

    avg_loss = running_loss / max(1, total_batches)
    avg_imp = running_impulse_loss / max(1, total_batches)
    avg_mrstft = running_mrstft_loss / max(1, total_batches)

    return {
        "train_loss": avg_loss,
        "train_impulse_loss": avg_imp,
        "train_mrstft_loss": avg_mrstft
    }


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    sr: int = 16000
) -> Dict[str, float]:
    """Evaluates validation loss, SI-SDR, SNR improvement, and STOI."""
    model.eval()
    running_loss = 0.0
    all_si_sdr = []
    all_snr_imp = []
    all_stoi = []

    with torch.no_grad():
        for batch in loader:
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)

            enhanced, _, _ = model(noisy)
            _, loss_dict = criterion(enhanced, clean, noisy)
            running_loss += loss_dict["total_loss"]

            metrics = evaluate_batch(enhanced, clean, noisy, sr=sr)
            all_si_sdr.append(metrics["si_sdr_db"])
            all_snr_imp.append(metrics["snr_improvement_db"])
            if "stoi" in metrics:
                all_stoi.append(metrics["stoi"])

    total_batches = max(1, len(loader))
    val_summary = {
        "val_loss": running_loss / total_batches,
        "val_si_sdr_db": sum(all_si_sdr) / max(1, len(all_si_sdr)),
        "val_snr_imp_db": sum(all_snr_imp) / max(1, len(all_snr_imp))
    }
    if all_stoi:
        val_summary["val_stoi"] = sum(all_stoi) / len(all_stoi)

    return val_summary


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_si_sdr: float,
    config: Config
):
    """Saves model weights and training state."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_si_sdr": best_si_sdr,
        "config": config
    }, path)


def train(config: Config, args: argparse.Namespace):
    """Main training loop."""
    device = torch.device(args.device if args.device else config.device)
    print(f"[+] Initializing AEGIS-ANC Training on device: {device}")

    # Build Model
    model = CausalFullSubNetPlus(config).to(device)
    num_params = model.get_num_parameters()
    print(f"[+] CausalFullSubNet+ Model built successfully: {num_params:,} parameters")

    # Build Loss
    train_cfg = config.train
    criterion = CompositeDefenceLoss(
        alpha=train_cfg.alpha_impulse,
        tau=train_cfg.tau_impulse,
        lambda_mrstft=train_cfg.lambda_mrstft
    ).to(device)

    # Build DataLoaders
    train_loader, val_loader = create_dataloaders(config=config)
    print(f"[+] DataLoaders ready: {len(train_loader)} train batches, {len(val_loader)} val batches")

    # Optimizer & LR Scheduler
    epochs = args.epochs if args.epochs else train_cfg.epochs
    lr = args.lr if args.lr else train_cfg.lr
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=train_cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else train_cfg.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_epoch = 1
    best_si_sdr = -999.0

    # Resume from checkpoint if specified
    if args.resume:
        if os.path.exists(args.resume):
            print(f"[+] Resuming checkpoint from: {args.resume}")
            chk = torch.save(args.resume, map_location=device)
            model.load_state_dict(chk["model_state_dict"])
            optimizer.load_state_dict(chk["optimizer_state_dict"])
            start_epoch = chk["epoch"] + 1
            best_si_sdr = chk.get("best_si_sdr", -999.0)
            print(f"[✓] Resumed at epoch {start_epoch} (Best SI-SDR: {best_si_sdr:.2f} dB)")
        else:
            print(f"[-] Checkpoint {args.resume} not found. Starting from scratch.")

    print("\n" + "=" * 70)
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'SI-SDR (dB)':^12} | {'ΔSNR (dB)':^10} | {'STOI':^8}")
    print("=" * 70)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        train_res = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_grad_norm=train_cfg.max_grad_norm
        )
        scheduler.step()

        val_res = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            sr=config.audio.sample_rate
        )
        elapsed = time.time() - t0

        stoi_str = f"{val_res.get('val_stoi', 0.0):.4f}" if "val_stoi" in val_res else "N/A"
        print(f"{epoch:^7d} | {train_res['train_loss']:^12.4f} | {val_res['val_loss']:^10.4f} | {val_res['val_si_sdr_db']:^12.2f} | {val_res['val_snr_imp_db']:^10.2f} | {stoi_str:^8} ({elapsed:.1f}s)")

        # Save latest checkpoint
        latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
        save_checkpoint(latest_path, model, optimizer, epoch, best_si_sdr, config)

        # Save best model
        current_si_sdr = val_res["val_si_sdr_db"]
        if current_si_sdr > best_si_sdr:
            best_si_sdr = current_si_sdr
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            save_checkpoint(best_path, model, optimizer, epoch, best_si_sdr, config)
            print(f"    [*] New best model saved! (SI-SDR: {best_si_sdr:.2f} dB) -> {best_path}")

    print("=" * 70)
    print(f"[+] Training Complete! Best Validation SI-SDR: {best_si_sdr:.2f} dB")


def main():
    parser = argparse.ArgumentParser(description="AEGIS-ANC Training Pipeline")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    config = Config()
    if args.batch_size:
        config.data.batch_size = args.batch_size

    train(config, args)


if __name__ == "__main__":
    main()
