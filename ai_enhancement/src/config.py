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
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
