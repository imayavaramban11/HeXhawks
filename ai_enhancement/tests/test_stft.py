import torch
from src.config import AudioConfig
from src.stft import STFT, ISTFT

def test_stft_istft_reconstruction():
    config = AudioConfig(center=False)
    stft = STFT(config)
    istft = ISTFT(config)
    
    # Create dummy waveform: Batch size 2, 16000 samples (1 second)
    batch_size = 2
    seq_len = 16000
    waveform = torch.randn(batch_size, seq_len)
    
    # Forward STFT
    complex_spec = stft(waveform)
    
    # Check dimensions
    # For n_fft=320, we expect F = 161 (n_fft//2 + 1)
    # For center=False, T_frames = 1 + (seq_len - win_length) // hop_length
    # 1 + (16000 - 320) // 160 = 1 + 15680 // 160 = 1 + 98 = 99
    assert complex_spec.shape == (batch_size, 161, 99)
    assert complex_spec.is_complex()
    
    # Backward iSTFT
    reconstructed = istft(complex_spec, length=seq_len)
    
    # Due to center=False and without perfect COLA windowing at the edges,
    # the exact reconstruction depends on the framing. With Hann window and 50% overlap,
    # interior frames reconstruct perfectly.
    
    # Let's check interior error (ignoring first and last frame boundaries)
    trim = config.n_fft
    error = torch.mean(torch.abs(waveform[:, trim:-trim] - reconstructed[:, trim:-trim]))
    
    # Error should be very small
    assert error < 1e-6
