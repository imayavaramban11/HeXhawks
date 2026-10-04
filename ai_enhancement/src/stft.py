import torch
import torch.nn as nn
from .config import AudioConfig

class STFT(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = config.win_length
        self.center = config.center
        
        # Create window
        if config.window_fn == "hann":
            window = torch.hann_window(self.win_length)
        else:
            window = torch.ones(self.win_length)
            
        self.register_buffer("window", window)
        
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: (B, T) or (T,)
        Returns: Complex tensor of shape (B, F, T_frames)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        complex_spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True
        )
        return complex_spec
        
class ISTFT(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = config.win_length
        self.center = config.center
        
        if config.window_fn == "hann":
            window = torch.hann_window(self.win_length)
        else:
            window = torch.ones(self.win_length)
            
        self.register_buffer("window", window)
        
    def forward(self, stft_matrix: torch.Tensor, length: int = None) -> torch.Tensor:
        """
        stft_matrix: Complex tensor of shape (B, F, T_frames)
        Returns: Waveform tensor of shape (B, T)
        """
        waveform = torch.istft(
            stft_matrix,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            normalized=False,
            onesided=True,
            length=None, # explicitly ignore length to avoid NOLA window overlap add min: 1 error for center=False
            return_complex=False
        )
        return waveform
