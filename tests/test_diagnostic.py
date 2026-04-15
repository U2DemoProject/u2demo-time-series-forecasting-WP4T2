"""Tests for u2.diagnostic module.

Verifies that sanity_check runs without errors and logs expected messages."""

import logging

import pytest

from u2.diagnostic import sanity_check


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
        monkeypatch.setattr("u2.diagnostic.torch.cuda.is_available", lambda: False)

        with caplog.at_level(logging.INFO, logger="U2 Time Series Prediction"):
            sanity_check()

        assert any("CUDA is not available" in record.message for record in caplog.records)
