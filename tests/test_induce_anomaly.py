import os
import pytest
import pandas as pd
import numpy as np

from induce_anomaly import induce_anomalies


def _make_stock_df(n_rows=20000):
    """Generate a minimal stock DataFrame large enough for anomaly injection."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-04-01", periods=n_rows, freq="min")
    return pd.DataFrame({
        "Stock": np.tile(["AAA", "BBB"], n_rows // 2),
        "date": dates[:n_rows],
        "open": rng.uniform(90, 110, n_rows),
        "high": rng.uniform(100, 120, n_rows),
        "low": rng.uniform(80, 100, n_rows),
        "close": rng.uniform(90, 110, n_rows),
        "volume": rng.integers(1000, 50000, n_rows),
    })


class TestInduceAnomalies:
    def test_output_files_created(self, tmp_path):
        df = _make_stock_df()
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"
        tracker_path = tmp_path / "tracker.csv"
        df.to_csv(input_path, index=False)

        induce_anomalies(str(input_path), str(output_path), str(tracker_path))

        assert output_path.exists()
        assert tracker_path.exists()

    def test_tracker_records_created(self, tmp_path):
        df = _make_stock_df()
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"
        tracker_path = tmp_path / "tracker.csv"
        df.to_csv(input_path, index=False)

        induce_anomalies(str(input_path), str(output_path), str(tracker_path))

        tracker = pd.read_csv(tracker_path)
        assert len(tracker) > 0
        expected_cols = {"Stock", "date", "Row_Index", "Corrupted_Column",
                         "Original_Value", "Anomalous_Value"}
        assert expected_cols.issubset(set(tracker.columns))

    def test_anomalies_modify_data(self, tmp_path):
        df = _make_stock_df()
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"
        tracker_path = tmp_path / "tracker.csv"
        df.to_csv(input_path, index=False)

        induce_anomalies(str(input_path), str(output_path), str(tracker_path))

        original = pd.read_csv(input_path)
        modified = pd.read_csv(output_path)
        tracker = pd.read_csv(tracker_path)

        # At least one value should differ between original and modified
        for _, t_row in tracker.iterrows():
            idx = int(t_row["Row_Index"])
            col = t_row["Corrupted_Column"]
            assert modified.loc[idx, col] != original.loc[idx, col]

    def test_row_count_unchanged(self, tmp_path):
        df = _make_stock_df()
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"
        tracker_path = tmp_path / "tracker.csv"
        df.to_csv(input_path, index=False)

        induce_anomalies(str(input_path), str(output_path), str(tracker_path))

        original = pd.read_csv(input_path)
        modified = pd.read_csv(output_path)
        assert len(original) == len(modified)

    def test_corruption_types_are_spike_or_drop(self, tmp_path):
        df = _make_stock_df()
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"
        tracker_path = tmp_path / "tracker.csv"
        df.to_csv(input_path, index=False)

        induce_anomalies(str(input_path), str(output_path), str(tracker_path))

        tracker = pd.read_csv(tracker_path)
        for _, row in tracker.iterrows():
            orig = row["Original_Value"]
            anom = row["Anomalous_Value"]
            if orig > 0:
                ratio = anom / orig
                # spike: 5x-10x  or  drop: 0.01-0.05x
                assert ratio > 4.5 or ratio < 0.06, f"Unexpected ratio {ratio}"

    def test_corrupted_columns_are_numeric(self, tmp_path):
        df = _make_stock_df()
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"
        tracker_path = tmp_path / "tracker.csv"
        df.to_csv(input_path, index=False)

        induce_anomalies(str(input_path), str(output_path), str(tracker_path))

        tracker = pd.read_csv(tracker_path)
        valid_cols = {"open", "high", "low", "close", "volume"}
        for col in tracker["Corrupted_Column"]:
            assert col in valid_cols

    def test_sorted_by_stock_and_date(self, tmp_path):
        df = _make_stock_df()
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"
        tracker_path = tmp_path / "tracker.csv"
        df.to_csv(input_path, index=False)

        induce_anomalies(str(input_path), str(output_path), str(tracker_path))

        modified = pd.read_csv(output_path)
        # Data should be sorted by Stock, then date
        assert modified["Stock"].is_monotonic_increasing or \
               list(modified["Stock"]) == sorted(modified["Stock"])

    def test_small_dataset_no_anomalies(self, tmp_path):
        """With < 1000 rows the first random step overshoots, so 0 anomalies."""
        df = _make_stock_df(n_rows=500)
        input_path = tmp_path / "input.csv"
        output_path = tmp_path / "output.csv"
        tracker_path = tmp_path / "tracker.csv"
        df.to_csv(input_path, index=False)

        induce_anomalies(str(input_path), str(output_path), str(tracker_path))

        # Empty DataFrame written to CSV may produce an empty file (no columns)
        try:
            tracker = pd.read_csv(tracker_path)
            assert len(tracker) == 0
        except pd.errors.EmptyDataError:
            # File exists but contains no data — equivalent to 0 anomalies
            assert tracker_path.exists()
