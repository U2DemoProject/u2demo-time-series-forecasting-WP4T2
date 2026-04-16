"""Tests for u2demo_time_series_forecasting.diagnostic module.

Verifies that sanity_check runs without errors and logs expected messages."""

import logging

import pytest

from u2demo_time_series_forecasting.diagnostic import sanity_check


@pytest.mark.unit
class TestSanityCheck:
    """Tests for sanity_check function."""

    def test_does_not_raise(self):
        """sanity_check should run without errors regardless of CUDA availability."""
        sanity_check()

    def test_logs_cuda_availability(self, caplog):
        """Should log whether CUDA is available."""
        with caplog.at_level(logging.INFO, logger="U2 Time Series Prediction"):
            sanity_check()

        assert any("CUDA" in record.message for record in caplog.records)

    def test_logs_cuda_not_available(self, monkeypatch, caplog):
        """When CUDA is not available, should log that."""
        monkeypatch.setattr("u2demo_time_series_forecasting.diagnostic.torch.cuda.is_available", lambda: False)

        with caplog.at_level(logging.INFO, logger="U2 Time Series Prediction"):
            sanity_check()

        assert any("CUDA is not available" in record.message for record in caplog.records)

    def test_logs_cuda_available_with_gpus(self, monkeypatch, caplog):
        """When CUDA is available, should log version and GPU details."""
        monkeypatch.setattr("u2demo_time_series_forecasting.diagnostic.torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("u2demo_time_series_forecasting.diagnostic.torch.version.cuda", "12.4")
        monkeypatch.setattr("u2demo_time_series_forecasting.diagnostic.torch.cuda.device_count", lambda: 2)
        monkeypatch.setattr(
            "u2demo_time_series_forecasting.diagnostic.torch.cuda.get_device_name", lambda i: f"FakeGPU-{i}"
        )

        with caplog.at_level(logging.INFO, logger="U2 Time Series Prediction"):
            sanity_check()

        messages = [r.message for r in caplog.records]
        assert any("CUDA is available" in m for m in messages)
        assert any("12.4" in m for m in messages)
        assert any("2" in m for m in messages)
        assert any("FakeGPU-0" in m for m in messages)
        assert any("FakeGPU-1" in m for m in messages)
