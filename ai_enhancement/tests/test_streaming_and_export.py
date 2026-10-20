"""
Unit Tests for Real-Time Streaming Engine and ONNX Edge Exporter.
"""

import os
import pytest
import torch
import onnx

from src.config import Config
from src.model import CausalFullSubNetPlus
from src.streaming import StreamingAudioBuffer, StreamingEnhancer
from export import export_model_to_onnx

def test_streaming_audio_buffer():
    buf = StreamingAudioBuffer(win_size=320, hop_size=160)
    assert buf.buffer.shape == (1, 320)
    assert torch.all(buf.buffer == 0.0)

    # Insert chunk of 1s
    chunk1 = torch.ones(1, 160)
    out1 = buf.update(chunk1)
    assert out1.shape == (1, 320)
    # First half 0, second half 1
    assert torch.all(out1[:, :160] == 0.0)
    assert torch.all(out1[:, 160:] == 1.0)

    # Insert chunk of 2s
    chunk2 = torch.ones(1, 160) * 2.0
    out2 = buf.update(chunk2)
    # First half 1, second half 2
    assert torch.all(out2[:, :160] == 1.0)
    assert torch.all(out2[:, 160:] == 2.0)

def test_streaming_enhancer_process_chunk():
    config = Config()
    model = CausalFullSubNetPlus(config)
    enhancer = StreamingEnhancer(model=model, config=config)

    chunk = torch.randn(1, 160)
    out_chunk, lat_ms = enhancer.process_chunk(chunk)

    assert out_chunk.shape == (1, 160)
    assert lat_ms > 0.0

def test_streaming_enhancer_latency_benchmark():
    config = Config()
    model = CausalFullSubNetPlus(config)
    enhancer = StreamingEnhancer(model=model, config=config)

    bench = enhancer.benchmark(num_frames=20)

    assert "mean_latency_ms" in bench
    assert "real_time_factor" in bench
    assert "meets_budget_20ms" in bench
    print(f"Latency: {bench['mean_latency_ms']:.2f} ms | RTF: {bench['real_time_factor']:.3f}")

    # Must be capable of real-time streaming
    assert bench["real_time_factor"] < 1.0
    assert bench["meets_budget_20ms"] is True

def test_streaming_process_stream():
    config = Config()
    model = CausalFullSubNetPlus(config)
    enhancer = StreamingEnhancer(model=model, config=config)

    # 1 second stream = 16000 samples
    stream = torch.randn(1, 16000)
    out_stream, telemetry = enhancer.process_stream(stream)

    assert out_stream.shape == (1, 16000)
    assert telemetry["chunks_processed"] == 100
    assert telemetry["is_real_time"] is True

def test_onnx_model_export(tmp_path):
    config = Config()
    model = CausalFullSubNetPlus(config)
    onnx_file = os.path.join(tmp_path, "aegis_test.onnx")

    export_model_to_onnx(model=model, output_path=onnx_file, opset_version=17)

    assert os.path.exists(onnx_file)
    size_bytes = os.path.getsize(onnx_file)
    # Model should be compact (< 2 MB)
    assert 100_000 < size_bytes < 2_000_000

    # Validate ONNX model with official onnx validator
    onnx_proto = onnx.load(onnx_file)
    onnx.checker.check_model(onnx_proto)
