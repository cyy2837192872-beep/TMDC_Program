"""Tests for core.io_utils — dataset and checkpoint loading."""

import os

import pytest


class TestRequireFile:
    def test_missing_raises(self):
        from angle_cnn.core.io_utils import require_file
        with pytest.raises(FileNotFoundError):
            require_file("/nonexistent/path/file.txt", "Test file")

    def test_empty_raises(self):
        from angle_cnn.core.io_utils import require_file
        with pytest.raises(FileNotFoundError):
            require_file("", "Test file")

    def test_existing_returns_path(self, tmp_path):
        from angle_cnn.core.io_utils import require_file
        f = tmp_path / "test.txt"
        f.write_text("hello")
        result = require_file(str(f), "Test")
        assert result == str(f)
