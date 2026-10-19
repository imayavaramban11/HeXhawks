from dataclasses import dataclass, field
import torch

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_fft: int = 320  # 20ms at 16kHz
    hop_length: int = 160 # 10ms at 16kHz
    win_length: int = 320
    window_fn: str = "hann"
    center: bool = True

@dataclass
class ModelConfig:
    num_freq_bins: int = 161      # n_fft // 2 + 1 for n_fft=320
    fullband_dim: int = 64        # full-band embedding dimension
    fullband_layers: int = 3      # number of causal TCN blocks
    subband_num_neighbors: int = 1 # 1 on each side -> window of 3 frequency bins
    subband_hidden_size: int = 128 # unidirectional LSTM hidden units
    subband_num_layers: int = 1   # strictly 1 layer unidirectional for streaming latency
    attention_reduction: int = 4  # reduction ratio for channel attention
    dropout: float = 0.0

@dataclass
class DataConfig:
    clean_dir: str = "ai_enhancement/data/clean_speech"
    noise_dir: str = "ai_enhancement/data/defence_noise"
    sample_rate: int = 16000
    segment_length: int = 48000     # 3.0 seconds at 16kHz
    min_snr_db: float = -5.0        # -5 dB extreme noise
    max_snr_db: float = 15.0        # +15 dB light noise
    impulse_probability: float = 0.5 # 50% chance of sudden gunshot/blast injection
    batch_size: int = 4
    num_workers: int = 0            # 0 is safe on Windows

@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
