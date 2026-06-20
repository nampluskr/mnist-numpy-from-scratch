# test_logger.py: Unit tests for Logger class.

import os
import pytest

from src.core.logger import Logger


class TestLogger:
    def test_append_increases_list_length(self):
        logger = Logger()
        logger.append(1, 1.0, 0.5)
        logger.append(2, 0.8, 0.6)
        assert len(logger.epochs) == 2
        assert len(logger.losses) == 2
        assert len(logger.metrics) == 2

    def test_load_epochs_count_matches(self):
        logs = [
            {"epoch": 1, "loss": 1.0, "metric": 0.5},
            {"epoch": 2, "loss": 0.8, "metric": 0.6},
            {"epoch": 3, "loss": 0.6, "metric": 0.7},
        ]
        logger = Logger()
        logger.load(logs)
        assert len(logger.epochs) == 3
        assert logger.epochs == [1, 2, 3]

    def test_to_dict_keys(self):
        logger = Logger()
        logger.append(1, 0.5, 0.8)
        d = logger.to_dict()
        assert set(d.keys()) == {"epochs", "losses", "metrics"}

    def test_to_csv_creates_file(self, tmp_path):
        logger = Logger()
        logger.append(1, 1.234, 0.512)
        logger.append(2, 0.891, 0.723)
        path = str(tmp_path / "log.csv")
        logger.to_csv(path)
        assert os.path.exists(path)
        with open(path) as f:
            lines = f.readlines()
        assert lines[0].strip() == "epoch,loss,metric"
        assert len(lines) == 3  # header + 2 data rows

    def test_empty_logger_to_dict(self):
        logger = Logger()
        d = logger.to_dict()
        assert d["epochs"] == []
        assert d["losses"] == []
        assert d["metrics"] == []
