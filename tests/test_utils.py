"""Tests for u2.utils module."""

from unittest.mock import Mock

import pytest
import requests

from u2.utils import is_retryable_error, parse_prefixed_int_columns


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


class TestParsePrefixedIntColumns:
    """Tests for parse_prefixed_int_columns function."""

    def test_parse_no_matching_columns(self):
        """Test parsing with no matching columns."""
        columns = ["roll_7", "roll_24"]
        result = parse_prefixed_int_columns(columns, "lag_")
        assert result == []

    def test_parse_empty_column_list(self):
        """Test parsing empty column list."""
        columns = []
        result = parse_prefixed_int_columns(columns, "lag_")
        assert result == []

    def test_parse(self):
        """Test parsing large integers."""
        columns = ["lag_0", "lag_168", "lag_672", "lag_1000"]
        result = parse_prefixed_int_columns(columns, "lag_")
        assert sorted(result) == [0, 168, 672, 1000]

    def test_parse_invalid_suffix_raises_error(self):
        """Test that non-integer suffix raises ValueError."""
        columns = ["lag_abc"]
        with pytest.raises(ValueError, match="invalid literal"):
            parse_prefixed_int_columns(columns, "lag_")
