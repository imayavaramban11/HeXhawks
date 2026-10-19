"""
Dynamic Data Pipeline & On-the-Fly Audio Mixer for Military Speech Enhancement.

Features:
1. Dynamic Online Mixing: Synthesizes noisy audio on-the-fly at random SNR (-5 dB to +15 dB).
2. Impulse Injection: Randomly injects gunfire/explosion blasts at arbitrary sample offsets.
3. Synthetic Generators: Provides built-in harmonic speech and defence noise generators
   for immediate testing and zero-file quickstarts.
4. PyTorch Dataset & DataLoader integration with deterministic splitting.
"""

import os
import glob
import random
from typing import List, Tuple, Dict, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader

from .config import Config, DataConfig
from .audio import load_audio, normalize_audio

def mix_audio_at_snr(
    clean: torch.Tensor,
    noise: torch.Tensor,
    snr_db: float,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Dynamically mixes clean speech with noise to achieve exact target SNR (dB).
    Formula:
        scale = RMS(clean) / (RMS(noise) * 10^(snr_db / 20))
        noisy = clean + scale * noise
    Returns:
        noisy: (1, T)
        scaled_noise: (1, T)
        actual_snr: float
    """
    if clean.dim() == 1:
        clean = clean.unsqueeze(0)
    if noise.dim() == 1:
        noise = noise.unsqueeze(0)

    # Align lengths to clean length
    T_clean = clean.shape[-1]
    T_noise = noise.shape[-1]

    if T_noise < T_clean:
        # Loop noise if shorter than clean speech
        repeat_factor = (T_clean // T_noise) + 1
        noise = noise.repeat(1, repeat_factor)[:, :T_clean]
    elif T_noise > T_clean:
        # Take random crop of noise
        start_idx = random.randint(0, T_noise - T_clean)
        noise = noise[:, start_idx:start_idx + T_clean]

    # Compute RMS energy
    rms_clean = torch.sqrt(torch.mean(clean ** 2) + eps)
    rms_noise = torch.sqrt(torch.mean(noise ** 2) + eps)

    # Scale factor for desired SNR
    scale_factor = rms_clean / (rms_noise * (10.0 ** (snr_db / 20.0)))
    scaled_noise = noise * scale_factor

    noisy = clean + scaled_noise

    # Prevent digital clipping (> 1.0)
    max_amp = max(torch.max(torch.abs(noisy)).item(), torch.max(torch.abs(clean)).item())
    if max_amp > 1.0:
        clean = clean / max_amp
        scaled_noise = scaled_noise / max_amp
        noisy = noisy / max_amp

    actual_snr = 10.0 * torch.log10(
        (torch.sum(clean ** 2) + eps) / (torch.sum(scaled_noise ** 2) + eps)
    ).item()

    return noisy, clean, actual_snr


def inject_impulsive_noise(
    noisy: torch.Tensor,
    impulse: torch.Tensor,
    offset: Optional[int] = None,
    gain_multiplier: float = 2.0
) -> Tuple[torch.Tensor, int]:
    """
    Injects a sudden high-energy gunshot/explosion blast into a noisy waveform.
    """
    if noisy.dim() == 1:
        noisy = noisy.unsqueeze(0)
    if impulse.dim() == 1:
        impulse = impulse.unsqueeze(0)

    T_noisy = noisy.shape[-1]
    T_impulse = impulse.shape[-1]

    if T_impulse > T_noisy:
        impulse = impulse[:, :T_noisy]
        T_impulse = T_noisy

    if offset is None:
        offset = random.randint(0, max(0, T_noisy - T_impulse))

    noisy_out = noisy.clone()
    peak_noisy = torch.max(torch.abs(noisy_out)).item() + 1e-4
    peak_impulse = torch.max(torch.abs(impulse)).item() + 1e-4

    # Scale impulse to deliver a sharp transient burst
    scaled_impulse = impulse * (peak_noisy / peak_impulse) * gain_multiplier

    end_idx = min(offset + T_impulse, T_noisy)
    impulse_slice_len = end_idx - offset
    noisy_out[:, offset:end_idx] += scaled_impulse[:, :impulse_slice_len]

    return noisy_out, offset


# =========================================================================
# Built-in Synthetic Audio Generators (Formant speech, engine rumble, gunshots)
# =========================================================================

def generate_synthetic_speech(duration_sec: float = 3.0, sr: int = 16000) -> torch.Tensor:
    """
    Synthesizes pseudo-speech using multi-formant harmonics and pitch modulation.
    """
    num_samples = int(duration_sec * sr)
    t = torch.linspace(0, duration_sec, num_samples)

    # Fundamental frequency varying around 120-160 Hz (human vocal pitch)
    f0 = 130.0 + 20.0 * torch.sin(2.0 * torch.pi * 1.5 * t)
    phase = torch.cumsum(2.0 * torch.pi * f0 / sr, dim=0)

    # Formant resonances (F1 ~ 500Hz, F2 ~ 1500Hz, F3 ~ 2500Hz)
    vocal = (
        0.5 * torch.sin(phase) +
        0.3 * torch.sin(phase * 4.0) +
        0.2 * torch.sin(phase * 12.0) +
        0.1 * torch.sin(phase * 20.0)
    )

    # Syllabic envelope modulation (speech rhythm ~ 3-4 Hz)
    envelope = torch.clamp(torch.sin(2.0 * torch.pi * 3.5 * t) ** 2, min=0.05, max=1.0)
    speech = vocal * envelope
    return speech.unsqueeze(0)


def generate_synthetic_defence_noise(
    duration_sec: float = 3.0,
    sr: int = 16000,
    noise_type: str = "engine"
) -> torch.Tensor:
    """
    Synthesizes defence noise:
    - 'engine': Low-frequency multi-harmonic diesel tank rumble (60Hz, 120Hz, 180Hz) + broadband rumble.
    - 'rotor': Helicopter rotor blade chop with periodic amplitude modulation.
    - 'gunshot': Sharp impulsive blast with high-energy shockwave and exponential decay.
    """
    num_samples = int(duration_sec * sr)
    t = torch.linspace(0, duration_sec, num_samples)

    if noise_type == "engine":
        # Tank engine rumble: low frequencies + noise
        engine = (
            0.6 * torch.sin(2.0 * torch.pi * 60.0 * t) +
            0.3 * torch.sin(2.0 * torch.pi * 120.0 * t) +
            0.2 * torch.sin(2.0 * torch.pi * 180.0 * t) +
            0.15 * torch.randn_like(t)
        )
        return engine.unsqueeze(0)

    elif noise_type == "rotor":
        # Helicopter blade wash: periodic low-frequency thumping
        blade_rate = 18.0  # 18 chops per second
        chop = torch.clamp(torch.sin(2.0 * torch.pi * blade_rate * t), min=0.0) ** 4
        wash = (0.7 * chop * torch.randn_like(t)) + 0.3 * torch.sin(2.0 * torch.pi * 90.0 * t)
        return wash.unsqueeze(0)

    elif noise_type == "gunshot":
        # Gunshot impulse: shockwave burst + exponential decay
        decay_rate = 40.0
        impulse = torch.exp(-decay_rate * t) * torch.randn_like(t)
        return impulse.unsqueeze(0)

    else:
        # White / ambient noise fallback
        return torch.randn(1, num_samples) * 0.2


# =========================================================================
# PyTorch Dataset
# =========================================================================

class DefenceSpeechDataset(Dataset):
    """
    PyTorch Dataset for Dynamic Defence Speech Enhancement Training.
    Pairs clean speech with defence noise at dynamic SNRs on every __getitem__ call.
    Automatically falls back to high-fidelity synthetic generation if local audio
    files are not present yet.
    """
    def __init__(
        self,
        clean_files: Optional[List[str]] = None,
        noise_files: Optional[List[str]] = None,
        impulse_files: Optional[List[str]] = None,
        config: Optional[DataConfig] = None,
        virtual_size: int = 128
    ):
        super().__init__()
        self.config = config or DataConfig()
        self.sr = self.config.sample_rate
        self.seg_len = self.config.segment_length
        self.min_snr = self.config.min_snr_db
        self.max_snr = self.config.max_snr_db
        self.p_impulse = self.config.impulse_probability
        self.virtual_size = virtual_size

        # Discover or assign files
        self.clean_files = clean_files or self._discover_files(self.config.clean_dir)
        self.noise_files = noise_files or self._discover_files(self.config.noise_dir)
        self.impulse_files = impulse_files or []

        self.use_synthetic = len(self.clean_files) == 0 or len(self.noise_files) == 0

    def _discover_files(self, directory: str) -> List[str]:
        if not os.path.exists(directory):
            return []
        patterns = ["*.wav", "*.flac", "*.mp3"]
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(directory, "**", p), recursive=True))
        return files

    def __len__(self) -> int:
        if self.use_synthetic:
            return self.virtual_size
        return len(self.clean_files)

    def _load_or_crop(self, path: str) -> torch.Tensor:
        audio = load_audio(path, target_sr=self.sr)
        T = audio.shape[-1]
        if T < self.seg_len:
            # Pad
            audio = torch.nn.functional.pad(audio, (0, self.seg_len - T))
        elif T > self.seg_len:
            start = random.randint(0, T - self.seg_len)
            audio = audio[:, start:start + self.seg_len]
        return audio

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, float, bool]]:
        # 1. Obtain Clean Speech
        if self.use_synthetic:
            clean = generate_synthetic_speech(duration_sec=self.seg_len / self.sr, sr=self.sr)
        else:
            clean_path = self.clean_files[idx % len(self.clean_files)]
            clean = self._load_or_crop(clean_path)

        # 2. Obtain Background Defence Noise
        if self.use_synthetic:
            noise_type = random.choice(["engine", "rotor"])
            noise = generate_synthetic_defence_noise(duration_sec=self.seg_len / self.sr, sr=self.sr, noise_type=noise_type)
        else:
            noise_path = random.choice(self.noise_files)
            noise = self._load_or_crop(noise_path)

        # 3. Dynamic Mixing with Random SNR
        snr_target = random.uniform(self.min_snr, self.max_snr)
        noisy, clean, actual_snr = mix_audio_at_snr(clean, noise, snr_target)

        # 4. Optional Sudden Gunshot / Blast Injection
        has_impulse = False
        if random.random() < self.p_impulse:
            has_impulse = True
            if self.impulse_files:
                impulse_path = random.choice(self.impulse_files)
                impulse = load_audio(impulse_path, target_sr=self.sr)
            else:
                impulse = generate_synthetic_defence_noise(duration_sec=0.2, sr=self.sr, noise_type="gunshot")

            noisy, _ = inject_impulsive_noise(noisy, impulse, gain_multiplier=2.5)

        return {
            "noisy": noisy.squeeze(0),      # (T,)
            "clean": clean.squeeze(0),      # (T,)
            "snr_db": actual_snr,
            "has_impulse": has_impulse
        }


def create_dataloaders(
    config: Optional[Config] = None,
    clean_files: Optional[List[str]] = None,
    noise_files: Optional[List[str]] = None,
    train_split: float = 0.8,
    virtual_size: int = 128
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates PyTorch Train and Validation DataLoaders.
    """
    cfg = config or Config()
    data_cfg = cfg.data

    if clean_files and len(clean_files) > 1:
        random.shuffle(clean_files)
        split_idx = int(len(clean_files) * train_split)
        train_clean = clean_files[:split_idx]
        val_clean = clean_files[split_idx:]
    else:
        train_clean = clean_files
        val_clean = clean_files

    train_dataset = DefenceSpeechDataset(
        clean_files=train_clean,
        noise_files=noise_files,
        config=data_cfg,
        virtual_size=int(virtual_size * train_split)
    )
    val_dataset = DefenceSpeechDataset(
        clean_files=val_clean,
        noise_files=noise_files,
        config=data_cfg,
        virtual_size=int(virtual_size * (1.0 - train_split))
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader
