import torch
import torchaudio
import soundfile as sf
import numpy as np

def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    Loads an audio file and resamples if necessary.
    Returns: Tensor of shape (C, T)
    """
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    
    # Ensure it's not silently changed to mono if it shouldn't be, but for this project we want mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    return waveform

def save_audio(path: str, waveform: torch.Tensor, sr: int = 16000):
    """
    Saves an audio waveform.
    waveform: Tensor of shape (C, T) or (T,)
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    # Use soundfile directly to avoid torchcodec dependency issues
    wav_np = waveform.squeeze(0).detach().cpu().numpy()
    sf.write(path, wav_np, sr)

def normalize_audio(waveform: torch.Tensor) -> torch.Tensor:
    """
    Peak normalizes the audio to [-1.0, 1.0].
    """
    max_val = torch.max(torch.abs(waveform))
    if max_val > 0:
        return waveform / max_val
    return waveform
