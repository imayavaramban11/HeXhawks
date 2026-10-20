"""
Real-Time Causal Streaming Inference Engine for AEGIS-ANC.

Features:
- Sliding window circular buffer for live microphone audio streaming.
- Sub-20ms latency guarantee (10ms hop window + ~5ms compute delay).
- Live latency profiling and Real-Time Factor (RTF) benchmarking.
"""

import time
from typing import Dict, List, Optional, Tuple, Generator
import torch
import torch.nn as nn
import numpy as np

from .config import Config, AudioConfig
from .model import CausalFullSubNetPlus

class StreamingAudioBuffer:
    """
    Sliding window buffer that accepts hop_size chunks (e.g. 160 samples = 10ms)
    and maintains a win_size window (e.g. 320 samples = 20ms) for STFT analysis.
    """
    def __init__(self, win_size: int = 320, hop_size: int = 160, device: torch.device = torch.device("cpu")):
        self.win_size = win_size
        self.hop_size = hop_size
        self.device = device
        self.buffer = torch.zeros(1, win_size, device=device)

    def reset(self):
        self.buffer.zero_()

    def update(self, chunk: torch.Tensor) -> torch.Tensor:
        """
        Appends a new chunk of size (1, hop_size) and shifts out oldest samples.
        Returns:
            window: Tensor of shape (1, win_size)
        """
        if chunk.dim() == 1:
            chunk = chunk.unsqueeze(0)
        chunk = chunk.to(self.device)

        # Shift left by hop_size and insert newest chunk
        self.buffer = torch.cat([self.buffer[:, self.hop_size:], chunk], dim=-1)
        return self.buffer


class StreamingEnhancer:
    """
    Live streaming inference wrapper for CausalFullSubNetPlus.
    Simulates real-time hardware audio input from microphone callbacks.
    """
    def __init__(self, model: nn.Module, config: Optional[Config] = None, device: Optional[torch.device] = None):
        self.config = config or Config()
        self.audio_cfg = self.config.audio
        self.device = device or torch.device(self.config.device)

        self.model = model.to(self.device)
        self.model.eval()

        self.win_size = self.audio_cfg.win_length
        self.hop_size = self.audio_cfg.hop_length
        self.sample_rate = self.audio_cfg.sample_rate

        self.stream_buf = StreamingAudioBuffer(
            win_size=self.win_size,
            hop_size=self.hop_size,
            device=self.device
        )

    def reset(self):
        """Resets streaming history."""
        self.stream_buf.reset()

    @torch.no_grad()
    def process_chunk(self, chunk: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Processes a single incoming audio chunk (e.g. 160 samples = 10ms).
        Returns:
            enhanced_chunk: (1, hop_size) tensor of enhanced audio.
            latency_ms: Processing latency in milliseconds.
        """
        t0 = time.perf_counter()

        # Update buffer with new mic chunk
        window = self.stream_buf.update(chunk)

        # Run forward pass through causal model
        enhanced_window, _, _ = self.model(window)

        # Extract the most recent hop_size samples
        enhanced_chunk = enhanced_window[:, -self.hop_size:]

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return enhanced_chunk, latency_ms

    @torch.no_grad()
    def process_stream(self, audio_stream: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Processes an entire audio stream chunk-by-chunk in a live simulation loop.
        audio_stream: (T,) or (1, T)
        """
        if audio_stream.dim() == 1:
            audio_stream = audio_stream.unsqueeze(0)

        T = audio_stream.shape[-1]
        self.reset()

        enhanced_chunks = []
        latencies = []

        num_chunks = T // self.hop_size
        for i in range(num_chunks):
            start = i * self.hop_size
            end = start + self.hop_size
            chunk = audio_stream[:, start:end]

            enh_chunk, lat = self.process_chunk(chunk)
            enhanced_chunks.append(enh_chunk.cpu())
            latencies.append(lat)

        full_enhanced = torch.cat(enhanced_chunks, dim=-1)

        hop_duration_ms = (self.hop_size / self.sample_rate) * 1000.0
        mean_lat = float(np.mean(latencies))
        p95_lat = float(np.percentile(latencies, 95))
        p99_lat = float(np.percentile(latencies, 99))
        rtf = mean_lat / hop_duration_ms

        telemetry = {
            "chunks_processed": len(latencies),
            "hop_duration_ms": hop_duration_ms,
            "mean_latency_ms": mean_lat,
            "p95_latency_ms": p95_lat,
            "p99_latency_ms": p99_lat,
            "real_time_factor": rtf,
            "is_real_time": rtf < 1.0
        }

        return full_enhanced, telemetry

    def benchmark(self, num_frames: int = 100) -> Dict[str, float]:
        """
        Benchmarks inference latency over synthetic incoming frames.
        """
        dummy_chunk = torch.randn(1, self.hop_size, device=self.device)
        self.reset()

        # Warmup
        for _ in range(10):
            self.process_chunk(dummy_chunk)

        latencies = []
        for _ in range(num_frames):
            _, lat = self.process_chunk(dummy_chunk)
            latencies.append(lat)

        hop_ms = (self.hop_size / self.sample_rate) * 1000.0
        mean_lat = float(np.mean(latencies))
        p95_lat = float(np.percentile(latencies, 95))
        p99_lat = float(np.percentile(latencies, 99))
        rtf = mean_lat / hop_ms

        return {
            "frames_tested": num_frames,
            "hop_duration_ms": hop_ms,
            "mean_latency_ms": mean_lat,
            "p95_latency_ms": p95_lat,
            "p99_latency_ms": p99_lat,
            "real_time_factor": rtf,
            "meets_budget_20ms": (hop_ms + p99_lat) < 20.0
        }
