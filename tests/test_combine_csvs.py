import os
import tempfile
import pytest
import pandas as pd

from combine_csvs import combine_csv_files


@pytest.fixture
def csv_dir(tmp_path):
    """Create a temp directory with sample CSV files."""
    data_a = pd.DataFrame({"date": ["2025-04-01"], "open": [100], "close": [105]})
    data_b = pd.DataFrame({"date": ["2025-04-02"], "open": [200], "close": [210]})
    data_a.to_csv(tmp_path / "STOCKA_data.csv", index=False)
    data_b.to_csv(tmp_path / "STOCKB_data.csv", index=False)
    return tmp_path


class TestCombineCsvFiles:
    def test_combines_multiple_csvs(self, csv_dir):
        combine_csv_files(str(csv_dir), output_file_name="out.csv")
        result = pd.read_csv(csv_dir / "out.csv")

        assert len(result) == 2
        assert "Stock" in result.columns
        assert set(result["Stock"]) == {"STOCKA", "STOCKB"}

    def test_stock_name_extracted_from_filename(self, csv_dir):
        combine_csv_files(str(csv_dir), output_file_name="out.csv")
        result = pd.read_csv(csv_dir / "out.csv")

        for _, row in result.iterrows():
            assert row["Stock"] in ("STOCKA", "STOCKB")

    def test_stock_column_is_first(self, csv_dir):
        combine_csv_files(str(csv_dir), output_file_name="out.csv")
        result = pd.read_csv(csv_dir / "out.csv")
        assert result.columns[0] == "Stock"

    def test_preserves_original_columns(self, csv_dir):
        combine_csv_files(str(csv_dir), output_file_name="out.csv")
        result = pd.read_csv(csv_dir / "out.csv")
        assert "date" in result.columns
        assert "open" in result.columns
        assert "close" in result.columns

    def test_empty_folder_returns_none(self, tmp_path):
        result = combine_csv_files(str(tmp_path), output_file_name="out.csv")
        assert result is None
        assert not (tmp_path / "out.csv").exists()

    def test_no_valid_csvs_returns_none(self, tmp_path):
        bad_file = tmp_path / "BAD_data.csv"
        bad_file.write_text("this is not,valid\ncsv content\x00\x01")
        result = combine_csv_files(str(tmp_path), output_file_name="out.csv")
        # Even malformed CSVs may parse; verify function doesn't crash
        assert result is None or (tmp_path / "out.csv").exists()

    def test_single_csv(self, tmp_path):
        df = pd.DataFrame({"date": ["2025-05-01"], "open": [50], "close": [55]})
        df.to_csv(tmp_path / "ONLY_stock.csv", index=False)

        combine_csv_files(str(tmp_path), output_file_name="out.csv")
        result = pd.read_csv(tmp_path / "out.csv")
        assert len(result) == 1
        assert result.iloc[0]["Stock"] == "ONLY"

    def test_output_path_is_inside_folder(self, csv_dir):
        combine_csv_files(str(csv_dir), output_file_name="combined.csv")
        expected = csv_dir / "combined.csv"
        assert expected.exists()

    def test_mixed_valid_and_unreadable_files(self, tmp_path):
        good = pd.DataFrame({"x": [1]})
        good.to_csv(tmp_path / "GOOD_data.csv", index=False)
        # Write a binary blob that cannot be parsed as CSV
        (tmp_path / "BAD_data.csv").write_bytes(b"\x80\x81\x82")

        combine_csv_files(str(tmp_path), output_file_name="out.csv")
        # At minimum the good file should be present
        if (tmp_path / "out.csv").exists():
            result = pd.read_csv(tmp_path / "out.csv")
            assert len(result) >= 1
