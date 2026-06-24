"""
Tests for fetch_itc_minute_data.py.

The source file has a pre-existing syntax error (line 114: missing colon on
``while``), so a normal ``import`` fails. We work around this by reading the
source, patching the syntax error in memory, and exec-ing only the functions
under test into a synthetic module namespace.
"""

import datetime
import os
import sys
import textwrap
import types
import pytest
from unittest.mock import MagicMock, patch


_SRC_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "fetch_itc_minute_data.py"
)


def _load_fetch_functions():
    """
    Read fetch_itc_minute_data.py source, fix the known syntax error,
    provide a mock kiteconnect package, and exec the module.
    Returns a namespace dict containing all module-level names.
    """
    with open(_SRC_PATH) as f:
        source = f.read()

    # Fix the known syntax error on line 114
    source = source.replace(
        "while current_date < end_dt\n",
        "while current_date < end_dt:\n",
    )
    # Fix f-string with nested quotes on line 166 (Python < 3.12 compat)
    source = source.replace(
        'print(f"job Ended at {datetime.datetime.strptime(START_DATE, "%Y-%m-%d %H:%M:%S")}")',
        'print(f"job Ended at {datetime.datetime.strptime(START_DATE, \'%Y-%m-%d %H:%M:%S\')}")',
    )

    # Create mock kiteconnect
    mock_kite_module = types.ModuleType("kiteconnect")
    mock_kite_class = MagicMock()
    mock_kite_module.KiteConnect = mock_kite_class
    sys.modules["kiteconnect"] = mock_kite_module

    ns = {"__name__": "fetch_itc_minute_data", "__file__": _SRC_PATH}

    with patch("builtins.open", MagicMock()):
        exec(compile(source, _SRC_PATH, "exec"), ns)

    return ns, mock_kite_class


class TestGetInstrumentToken:
    def test_returns_token_for_matching_symbol(self):
        ns, mock_kite_cls = _load_fetch_functions()

        mock_kite_instance = MagicMock()
        mock_kite_cls.return_value = mock_kite_instance
        mock_kite_instance.instruments.return_value = [
            {"tradingsymbol": "ITC", "exchange": "NSE",
             "instrument_type": "EQ", "instrument_token": 438881},
            {"tradingsymbol": "TCS", "exchange": "NSE",
             "instrument_type": "EQ", "instrument_token": 123456},
        ]

        token = ns["get_instrument_token"]("ITC")
        assert token == 438881

    def test_returns_none_for_missing_symbol(self):
        ns, mock_kite_cls = _load_fetch_functions()

        mock_kite_instance = MagicMock()
        mock_kite_cls.return_value = mock_kite_instance
        mock_kite_instance.instruments.return_value = [
            {"tradingsymbol": "TCS", "exchange": "NSE",
             "instrument_type": "EQ", "instrument_token": 123456},
        ]

        token = ns["get_instrument_token"]("NONEXISTENT")
        assert token is None

    def test_returns_none_on_api_error(self):
        ns, mock_kite_cls = _load_fetch_functions()

        mock_kite_instance = MagicMock()
        mock_kite_cls.return_value = mock_kite_instance
        mock_kite_instance.instruments.side_effect = Exception("API error")

        token = ns["get_instrument_token"]("ITC")
        assert token is None

    def test_filters_by_exchange_and_type(self):
        ns, mock_kite_cls = _load_fetch_functions()

        mock_kite_instance = MagicMock()
        mock_kite_cls.return_value = mock_kite_instance
        mock_kite_instance.instruments.return_value = [
            {"tradingsymbol": "ITC", "exchange": "BSE",
             "instrument_type": "EQ", "instrument_token": 999},
            {"tradingsymbol": "ITC", "exchange": "NSE",
             "instrument_type": "FUT", "instrument_token": 888},
            {"tradingsymbol": "ITC", "exchange": "NSE",
             "instrument_type": "EQ", "instrument_token": 438881},
        ]

        token = ns["get_instrument_token"]("ITC")
        assert token == 438881


class TestFetchHistoricalData:
    def test_returns_data_on_success(self):
        ns, _ = _load_fetch_functions()

        mock_kite = MagicMock()
        expected = [
            {"date": "2025-04-01", "open": 100, "high": 105,
             "low": 95, "close": 102, "volume": 5000}
        ]
        mock_kite.historical_data.return_value = expected

        result = ns["fetch_historical_data"](
            mock_kite, 438881, "2025-04-01", "2025-04-02", "minute"
        )
        assert result == expected

    def test_returns_none_on_error(self):
        ns, _ = _load_fetch_functions()

        mock_kite = MagicMock()
        mock_kite.historical_data.side_effect = Exception("rate limit")

        result = ns["fetch_historical_data"](
            mock_kite, 438881, "2025-04-01", "2025-04-02", "minute"
        )
        assert result is None

    def test_passes_correct_params(self):
        ns, _ = _load_fetch_functions()

        mock_kite = MagicMock()
        mock_kite.historical_data.return_value = []

        ns["fetch_historical_data"](
            mock_kite, 438881, "2025-04-01", "2025-04-02", "minute"
        )

        mock_kite.historical_data.assert_called_once_with(
            instrument_token=438881,
            from_date="2025-04-01",
            to_date="2025-04-02",
            interval="minute",
            continuous=False,
            oi=False,
        )


class TestGetAccessToken:
    def test_returns_token_on_success(self):
        ns, mock_kite_cls = _load_fetch_functions()

        mock_kite_instance = MagicMock()
        mock_kite_cls.return_value = mock_kite_instance
        mock_kite_instance.login_url.return_value = "https://example.com/login"
        mock_kite_instance.generate_session.return_value = {
            "access_token": "test_token_123"
        }

        with patch("builtins.input", return_value="fake_request_token"):
            token = ns["get_access_token"]()

        assert token == "test_token_123"

    def test_returns_none_on_session_error(self):
        ns, mock_kite_cls = _load_fetch_functions()

        mock_kite_instance = MagicMock()
        mock_kite_cls.return_value = mock_kite_instance
        mock_kite_instance.login_url.return_value = "https://example.com/login"
        mock_kite_instance.generate_session.side_effect = Exception("Invalid token")

        with patch("builtins.input", return_value="bad_token"):
            token = ns["get_access_token"]()

        assert token is None


class TestModuleConstants:
    def test_trading_symbol_list_not_empty(self):
        ns, _ = _load_fetch_functions()
        assert len(ns["TRADING_SYMBOL_LIST"]) > 0

    def test_date_format_valid(self):
        ns, _ = _load_fetch_functions()
        datetime.datetime.strptime(ns["START_DATE"], "%Y-%m-%d %H:%M:%S")
        datetime.datetime.strptime(ns["END_DATE"], "%Y-%m-%d %H:%M:%S")

    def test_exchange_is_nse(self):
        ns, _ = _load_fetch_functions()
        assert ns["EXCHANGE"] == "NSE"
