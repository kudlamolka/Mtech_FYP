import pytest
import pandas as pd
import numpy as np

from Forest_Isolation_Batch import (
    load_and_clean_data,
    compute_market_and_timeline_features,
    process_anomalies_with_optimal_threshold,
    validate_against_tracker,
)


def _make_raw_df(n_stocks=3, days=60):
    """Synthetic minute-level stock data within the expected date range."""
    rng = np.random.default_rng(0)
    stocks = [f"STOCK{i}" for i in range(n_stocks)]
    records = []
    for stock in stocks:
        dates = pd.date_range("2025-04-01", periods=days, freq="h")
        for d in dates:
            records.append({
                "Stock": stock,
                "date": d,
                "open": rng.uniform(90, 110),
                "high": rng.uniform(100, 120),
                "low": rng.uniform(80, 100),
                "close": rng.uniform(90, 110),
                "volume": int(rng.integers(1000, 50000)),
            })
    return pd.DataFrame(records)


# --------------- load_and_clean_data ---------------

class TestLoadAndCleanData:
    def test_filters_to_date_range(self, tmp_path):
        df = _make_raw_df()
        # Add rows outside range
        extra = pd.DataFrame({
            "Stock": ["X"], "date": ["2020-01-01"],
            "open": [1], "high": [1], "low": [1], "close": [1], "volume": [1],
        })
        full = pd.concat([df, extra], ignore_index=True)
        path = tmp_path / "data.csv"
        full.to_csv(path, index=False)

        result = load_and_clean_data(str(path))
        assert result["date"].min() >= pd.Timestamp("2025-04-01")
        assert result["date"].max() <= pd.Timestamp("2026-05-31")

    def test_drops_rows_with_nan_ohlcv(self, tmp_path):
        df = _make_raw_df(n_stocks=1, days=10)
        df.loc[0, "close"] = np.nan
        path = tmp_path / "data.csv"
        df.to_csv(path, index=False)

        result = load_and_clean_data(str(path))
        assert result["close"].isna().sum() == 0

    def test_drops_rows_with_bad_dates(self, tmp_path):
        df = _make_raw_df(n_stocks=1, days=10)
        path = tmp_path / "data.csv"
        df.to_csv(path, index=False)
        # Inject a bad date string directly in the CSV
        text = path.read_text()
        lines = text.split("\n")
        # Corrupt the date field of the second data row (index 1)
        parts = lines[2].split(",")
        parts[1] = "NOT_A_DATE"  # 'date' is the second column
        lines[2] = ",".join(parts)
        path.write_text("\n".join(lines))

        result = load_and_clean_data(str(path))
        assert result["date"].isna().sum() == 0

    def test_sorted_by_date_and_stock(self, tmp_path):
        df = _make_raw_df()
        path = tmp_path / "data.csv"
        df.to_csv(path, index=False)

        result = load_and_clean_data(str(path))
        dates = result["date"].tolist()
        assert dates == sorted(dates)

    def test_returns_dataframe(self, tmp_path):
        df = _make_raw_df(n_stocks=1, days=5)
        path = tmp_path / "data.csv"
        df.to_csv(path, index=False)

        result = load_and_clean_data(str(path))
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0


# --------------- compute_market_and_timeline_features ---------------

class TestComputeFeatures:
    def test_adds_expected_columns(self):
        df = _make_raw_df(n_stocks=2, days=40)
        result = compute_market_and_timeline_features(df)

        expected_cols = [
            "return_1m", "return_5m", "hl_spread_pct",
            "volume_zscore", "price_dev_pct",
            "market_avg_return", "market_avg_volatility",
            "return_vs_market", "volatility_vs_market",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_nans_after_fill(self):
        df = _make_raw_df(n_stocks=2, days=40)
        result = compute_market_and_timeline_features(df)
        assert result[["return_1m", "return_5m", "hl_spread_pct",
                        "volume_zscore", "price_dev_pct"]].isna().sum().sum() == 0

    def test_return_1m_calculation(self):
        df = _make_raw_df(n_stocks=1, days=40)
        result = compute_market_and_timeline_features(df)
        # return_1m should be close to pct_change of close within each stock
        stock_data = result[result["Stock"] == "STOCK0"].reset_index(drop=True)
        manual = stock_data["close"].pct_change()
        # After fillna(0) applied by function, compare non-NaN positions
        for i in range(1, min(10, len(stock_data))):
            np.testing.assert_almost_equal(
                stock_data.iloc[i]["return_1m"],
                manual.iloc[i],
                decimal=5,
            )

    def test_hl_spread_pct_positive(self):
        df = _make_raw_df(n_stocks=1, days=20)
        # Ensure high > low > 0
        df["high"] = df["open"] + 10
        df["low"] = df["open"] - 5
        result = compute_market_and_timeline_features(df)
        assert (result["hl_spread_pct"] >= 0).all()

    def test_cross_sectional_features_relative(self):
        df = _make_raw_df(n_stocks=3, days=40)
        result = compute_market_and_timeline_features(df)
        # return_vs_market = return_1m - market_avg_return
        diff = (result["return_1m"] - result["market_avg_return"]) - result["return_vs_market"]
        np.testing.assert_array_almost_equal(diff.values, 0, decimal=10)

    def test_row_count_preserved(self):
        df = _make_raw_df(n_stocks=2, days=20)
        result = compute_market_and_timeline_features(df)
        assert len(result) == len(df)


# --------------- validate_against_tracker ---------------

class TestValidateAgainstTracker:
    def _make_results_df(self):
        """Build a minimal results DataFrame with is_anomaly & actual_anomaly."""
        return pd.DataFrame({
            "Stock": ["A"] * 10,
            "date": pd.date_range("2025-04-01", periods=10, freq="h"),
            "is_anomaly": [1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
            "actual_anomaly": [1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        })

    def test_runs_without_error(self, tmp_path):
        results = self._make_results_df()
        tracker = pd.DataFrame({
            "Stock": ["A", "A", "A"],
            "date": results["date"].iloc[[0, 5, 6]],
        })
        tracker_path = tmp_path / "tracker.csv"
        tracker.to_csv(tracker_path, index=False)

        # Should not raise
        validate_against_tracker(results, str(tracker_path))

    def test_handles_missing_actual_anomaly_column(self, tmp_path):
        results = self._make_results_df().drop(columns=["actual_anomaly"])
        tracker = pd.DataFrame({
            "Stock": ["A"],
            "date": [results["date"].iloc[0]],
        })
        tracker_path = tmp_path / "tracker.csv"
        tracker.to_csv(tracker_path, index=False)

        validate_against_tracker(results, str(tracker_path))

    def test_precision_and_recall_logic(self, capsys, tmp_path):
        results = self._make_results_df()
        tracker = pd.DataFrame({
            "Stock": ["A", "A", "A"],
            "date": results["date"].iloc[[0, 5, 6]],
        })
        tracker_path = tmp_path / "tracker.csv"
        tracker.to_csv(tracker_path, index=False)

        validate_against_tracker(results, str(tracker_path))
        captured = capsys.readouterr().out

        # TP=2 (rows 0,5 both predicted & actual), FP=1 (row 1), FN=1 (row 6)
        assert "True Positives" in captured
        assert "Precision" in captured
        assert "Recall" in captured


# --------------- process_anomalies_with_optimal_threshold ---------------

class TestProcessAnomalies:
    def test_end_to_end_small_dataset(self, tmp_path):
        rng = np.random.default_rng(42)
        n = 500
        df = pd.DataFrame({
            "Stock": np.tile(["A", "B"], n // 2),
            "date": pd.date_range("2025-04-01", periods=n, freq="h"),
            "open": rng.uniform(90, 110, n),
            "high": rng.uniform(100, 120, n),
            "low": rng.uniform(80, 100, n),
            "close": rng.uniform(90, 110, n),
            "volume": rng.integers(1000, 50000, n),
            "return_1m": rng.normal(0, 0.01, n),
            "return_5m": rng.normal(0, 0.02, n),
            "hl_spread_pct": rng.uniform(0, 0.1, n),
            "volume_zscore": rng.normal(0, 1, n),
            "price_dev_pct": rng.normal(0, 0.01, n),
            "return_vs_market": rng.normal(0, 0.005, n),
            "volatility_vs_market": rng.normal(0, 0.005, n),
        })

        # Create tracker with a handful of known anomaly rows
        anomaly_indices = [10, 50, 100, 200, 300]
        tracker = df.loc[anomaly_indices, ["Stock", "date"]].copy()
        tracker_path = tmp_path / "tracker.csv"
        tracker.to_csv(tracker_path, index=False)

        result = process_anomalies_with_optimal_threshold(df, str(tracker_path))

        assert "is_anomaly" in result.columns
        assert "raw_anomaly_score" in result.columns
        assert "actual_anomaly" in result.columns
        assert set(result["is_anomaly"].unique()).issubset({0, 1})
        assert len(result) == n
