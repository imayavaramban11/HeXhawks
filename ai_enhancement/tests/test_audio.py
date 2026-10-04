import torch
import os
from src.audio import load_audio, save_audio, normalize_audio

def test_audio_pipeline(tmp_path):
    # Generate dummy waveform
    sample_rate = 16000
    duration = 1.0 # seconds
    t = torch.linspace(0, duration, int(sample_rate * duration))
    waveform = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0) # 440Hz tone
    waveform = waveform * 0.5 # half amplitude
    
    # Save
    test_file = os.path.join(tmp_path, "test_audio.wav")
    save_audio(test_file, waveform, sample_rate)
    
    assert os.path.exists(test_file)
    
    # Load
    loaded_waveform = load_audio(test_file, target_sr=16000)
    
    assert loaded_waveform.shape == waveform.shape
    # Check max difference (should be small due to 16-bit PCM quantization)
    assert torch.max(torch.abs(waveform - loaded_waveform)) < 1e-3
    
    # Normalize
    norm_waveform = normalize_audio(loaded_waveform)
    assert torch.isclose(torch.max(torch.abs(norm_waveform)), torch.tensor(1.0))
