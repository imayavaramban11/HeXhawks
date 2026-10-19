import torch
from src.config import AudioConfig
from src.stft import STFT, ISTFT

def test_stft_istft_reconstruction():
    config = AudioConfig(center=True)
    stft = STFT(config)
    istft = ISTFT(config)
    
    # Create dummy waveform: Batch size 2, 16000 samples (1 second)
    batch_size = 2
    seq_len = 16000
    waveform = torch.randn(batch_size, seq_len)
    
    # Forward STFT
    complex_spec = stft(waveform)
    
    # For n_fft=320, we expect F = 161 (n_fft//2 + 1)
    # For center=True, T_frames = 1 + seq_len // hop_length = 1 + 16000 // 160 = 101
    assert complex_spec.shape == (batch_size, 161, 101)
    assert complex_spec.is_complex()
    
    # Backward iSTFT
    reconstructed = istft(complex_spec, length=seq_len)
    assert reconstructed.shape == (batch_size, seq_len)
    
    # Check reconstruction error across the entire waveform
    error = torch.mean(torch.abs(waveform - reconstructed))
    assert error < 1e-5
