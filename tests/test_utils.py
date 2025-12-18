"""Tests for u2.utils module."""

from unittest.mock import Mock

import requests

from u2.utils import is_retryable_error


class TestIsRetryableError:
    """Tests for is_retryable_error function."""

    def test_retryable_504_error(self):
        """Test that 504 Gateway Timeout is retryable."""
        exc = requests.exceptions.RequestException()
        exc.response = Mock(status_code=504)
        assert is_retryable_error(exc) is True

    def test_retryable_429_error(self):
        """Test that 429 Too Many Requests is retryable."""
        exc = requests.exceptions.RequestException()
        exc.response = Mock(status_code=429)
        assert is_retryable_error(exc) is True

    def test_non_retryable_error_status(self):
        """Test that other status codes are not retryable."""
        exc = requests.exceptions.RequestException()
        exc.response = Mock(status_code=500)
        assert is_retryable_error(exc) is False

    def test_non_request_exception(self):
        """Test that non-RequestException errors are not retryable."""
        exc = ValueError("Some error")
        assert is_retryable_error(exc) is False

    def test_exception_without_response(self):
        """Test that exceptions without response attribute are not retryable."""
        exc = requests.exceptions.RequestException()
        assert is_retryable_error(exc) is False

    def test_exception_with_none_response(self):
        """Test that exceptions with None response are not retryable."""
        exc = requests.exceptions.RequestException()
        exc.response = None
        assert is_retryable_error(exc) is False
