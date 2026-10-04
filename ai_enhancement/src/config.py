from dataclasses import dataclass, field
import torch

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_fft: int = 320  # 20ms at 16kHz
    hop_length: int = 160 # 10ms at 16kHz
    win_length: int = 320
    window_fn: str = "hann"
    center: bool = False # Must be False for streaming/causal STFT

@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
