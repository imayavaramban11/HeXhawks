"""
Causal FullSubNet+ Model for Military Speech Enhancement.
Combines:
1. Complex STFT representation (Magnitude, Real, Imaginary)
2. Causal Full-Band TCN + Channel Attention (global context)
3. Causal Unidirectional Sub-Band LSTM (local frequency details)
4. Complex Ideal Ratio Mask (cIRM) prediction & multiplication
5. Inverse STFT waveform reconstruction
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn

from .config import Config, ModelConfig, AudioConfig
from .stft import STFT, ISTFT
from .modules import (
    CausalFullBandExtractor,
    SubBandFeatureExtractor,
    apply_complex_mask
)

class CausalFullSubNetPlus(nn.Module):
    """
    Causal FullSubNet+ Neural Architecture for Real-Time Defence ANC & Speech Enhancement.
    """
    def __init__(self, config: Optional[Config] = None):
        super().__init__()
        self.config = config or Config()
        self.audio_cfg = self.config.audio
        self.model_cfg = self.config.model

        # STFT & iSTFT modules
        self.stft = STFT(self.audio_cfg)
        self.istft = ISTFT(self.audio_cfg)

        F_bins = self.model_cfg.num_freq_bins
        fullband_dim = self.model_cfg.fullband_dim
        num_neighbors = self.model_cfg.subband_num_neighbors

        # Full-Band branch: input has 3 channels (Mag, Real, Imag) * F_bins
        self.fullband_extractor = CausalFullBandExtractor(
            in_features=3 * F_bins,
            hidden_dim=fullband_dim,
            num_layers=self.model_cfg.fullband_layers,
            reduction=self.model_cfg.attention_reduction,
            dropout=self.model_cfg.dropout
        )

        # Sub-Band feature extractor: 3 channels * (2 * num_neighbors + 1)
        self.subband_extractor = SubBandFeatureExtractor(num_neighbors=num_neighbors)
        subband_feature_dim = 3 * (2 * num_neighbors + 1)

        # Total input to Sub-Band LSTM = subband_feature_dim + fullband_dim
        subband_in_dim = subband_feature_dim + fullband_dim

        # Strictly unidirectional LSTM for real-time causality
        self.subband_lstm = nn.LSTM(
            input_size=subband_in_dim,
            hidden_size=self.model_cfg.subband_hidden_size,
            num_layers=self.model_cfg.subband_num_layers,
            batch_first=True,
            bidirectional=False
        )

        # Linear projection to complex mask components (Real, Imag)
        self.mask_proj = nn.Linear(self.model_cfg.subband_hidden_size, 2)

    def forward_spec(self, noisy_spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes complex STFT directly.
        noisy_spec: Complex tensor of shape (B, F, T)
        Returns:
            enhanced_spec: Complex tensor of shape (B, F, T)
            mask: Float tensor of shape (B, F, T, 2)
        """
        B, F_dim, T = noisy_spec.shape

        # 1. Extract Real, Imaginary, and Magnitude
        x_r = noisy_spec.real
        x_i = noisy_spec.imag
        x_mag = torch.abs(noisy_spec)

        # (B, 3, F, T)
        feats_2d = torch.stack([x_mag, x_r, x_i], dim=1)

        # 2. Full-Band Processing
        # Flatten (3, F) into single feature dimension -> (B, 3*F, T)
        fb_in = feats_2d.view(B, 3 * F_dim, T)
        fb_emb = self.fullband_extractor(fb_in)  # (B, fullband_dim, T)

        # 3. Sub-Band Feature Extraction
        # (B, F, T, subband_feature_dim)
        sb_feats = self.subband_extractor(feats_2d)

        # Expand full-band embedding to all frequency bins: (B, 1, T, fullband_dim) -> (B, F, T, fullband_dim)
        fb_emb_expanded = fb_emb.permute(0, 2, 1).unsqueeze(1).expand(-1, F_dim, -1, -1)

        # Concatenate local sub-band and global full-band features: (B, F, T, subband_in_dim)
        sb_in = torch.cat([sb_feats, fb_emb_expanded], dim=-1)

        # 4. Sub-Band Unidirectional LSTM
        # Reshape to (B * F, T, subband_in_dim) for batched temporal processing
        sb_in_flat = sb_in.view(B * F_dim, T, -1)
        lstm_out, _ = self.subband_lstm(sb_in_flat)  # (B * F, T, hidden_size)

        # 5. Predict Complex Mask (Real, Imag)
        mask_flat = self.mask_proj(lstm_out)  # (B * F, T, 2)
        mask = mask_flat.view(B, F_dim, T, 2)  # (B, F, T, 2)

        # 6. Apply Complex Multiplication: S = X * M
        enhanced_spec = apply_complex_mask(noisy_spec, mask)

        return enhanced_spec, mask

    def forward(self, noisy_waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        End-to-end waveform processing.
        noisy_waveform: (B, T_audio) or (T_audio,)
        Returns:
            enhanced_waveform: (B, T_audio)
            mask: (B, F, T_frames, 2)
            enhanced_spec: (B, F, T_frames) complex tensor
        """
        if noisy_waveform.dim() == 1:
            noisy_waveform = noisy_waveform.unsqueeze(0)

        audio_length = noisy_waveform.shape[-1]

        # Forward STFT
        noisy_spec = self.stft(noisy_waveform)

        # Complex enhancement
        enhanced_spec, mask = self.forward_spec(noisy_spec)

        # Inverse STFT to reconstruct waveform
        enhanced_waveform = self.istft(enhanced_spec, length=audio_length)

        return enhanced_waveform, mask, enhanced_spec

    def get_num_parameters(self) -> int:
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
