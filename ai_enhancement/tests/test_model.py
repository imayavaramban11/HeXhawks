"""
Unit Tests for Causal FullSubNet+ Architecture & Neural Modules.
Tests:
- Strict Causality (Future samples do NOT affect past outputs)
- Shape Invariance
- Complex Mask Arithmetic
- Backpropagation & Gradient Flow
- Edge Model Footprint
"""

import pytest
import torch
from src.config import Config, ModelConfig, AudioConfig
from src.modules import (
    CausalConv1d,
    ChannelAttention1d,
    CausalTCNBlock,
    CausalFullBandExtractor,
    SubBandFeatureExtractor,
    apply_complex_mask
)
from src.model import CausalFullSubNetPlus

def test_causal_conv1d_strict_causality():
    """
    Mathematical proof of causality:
    If we modify the input at time step t_future, the output at t <= t_current
    MUST NOT CHANGE AT ALL.
    """
    conv = CausalConv1d(in_channels=16, out_channels=16, kernel_size=3, dilation=2)
    conv.eval()

    # Original input of length 50
    x_orig = torch.randn(1, 16, 50)
    out_orig = conv(x_orig)

    # Modified input: only modify from frame 30 to 49
    x_mod = x_orig.clone()
    x_mod[:, :, 30:] += torch.randn(1, 16, 20) * 5.0
    out_mod = conv(x_mod)

    # Check that outputs up to frame 29 are BIT-EXACT identical
    past_diff = torch.max(torch.abs(out_orig[:, :, :30] - out_mod[:, :, :30])).item()
    assert past_diff == 0.0, f"Causality violation! Future modification leaked to past with diff {past_diff}"

def test_channel_attention():
    attn = ChannelAttention1d(channels=32, reduction=4)
    x = torch.randn(2, 32, 40)
    out = attn(x)
    assert out.shape == (2, 32, 40)

def test_causal_tcn_block():
    block = CausalTCNBlock(channels=32, kernel_size=3, dilation=2)
    x = torch.randn(2, 32, 50)
    out = block(x)
    assert out.shape == (2, 32, 50)

def test_subband_extractor():
    extractor = SubBandFeatureExtractor(num_neighbors=1)
    # B=2, C=3 (mag, real, imag), F=161, T=40
    x = torch.randn(2, 3, 161, 40)
    out = extractor(x)
    # Expected output: (B, F, T, C * (2*K + 1)) -> 3 * 3 = 9
    assert out.shape == (2, 161, 40, 9)

def test_apply_complex_mask():
    # Test complex multiplication: X = 2 + 3j, M = 1 - 2j
    # X * M = (2*1 - 3*(-2)) + j*(2*(-2) + 3*1) = (2 + 6) + j*(-4 + 3) = 8 - 1j
    x = torch.complex(torch.tensor([[[2.0]]]), torch.tensor([[[3.0]]]))
    mask = torch.tensor([[[[1.0, -2.0]]]])  # Real=1, Imag=-2

    s_hat = apply_complex_mask(x, mask)
    assert torch.isclose(s_hat.real, torch.tensor([[[8.0]]]))
    assert torch.isclose(s_hat.imag, torch.tensor([[[ -1.0]]]))

def test_causal_fullsubnet_forward_and_backward():
    """
    Verifies end-to-end forward pass on raw 16kHz waveform and backward gradient flow.
    """
    config = Config()
    model = CausalFullSubNetPlus(config)
    model.train()

    # 1 second of audio at 16kHz for 2 batches
    batch_size = 2
    audio_len = 16000
    noisy_wav = torch.randn(batch_size, audio_len, requires_grad=True)

    enhanced_wav, mask, enhanced_spec = model(noisy_wav)

    # Check output shapes
    assert enhanced_wav.shape == (batch_size, audio_len)
    assert enhanced_spec.shape[0] == batch_size
    assert enhanced_spec.shape[1] == config.model.num_freq_bins
    assert mask.shape[-1] == 2  # (Real, Imag) mask

    # Backward pass test
    loss = torch.mean(torch.abs(enhanced_wav))
    loss.backward()

    # Verify gradients computed across subband LSTM and fullband TCN
    assert model.subband_lstm.weight_ih_l0.grad is not None
    assert model.fullband_extractor.input_proj.weight.grad is not None

def test_edge_model_parameter_count():
    """
    Ensures model is lightweight enough for Jetson edge deployment (< 1M parameters).
    """
    config = Config()
    model = CausalFullSubNetPlus(config)
    num_params = model.get_num_parameters()
    print(f"Total trainable parameters: {num_params:,}")

    # Should be under 1 million parameters for sub-20ms edge execution
    assert num_params < 1_000_000, f"Model has {num_params} parameters, exceeding edge budget!"
