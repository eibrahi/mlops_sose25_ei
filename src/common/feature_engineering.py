import numpy as np
import pandas as pd


OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
LAG_PERIODS = [1, 2, 3, 5, 10, 20, 50]
RETURN_PERIODS = [1, 5, 10, 20]
ROLLING_WINDOWS = [5, 10, 20, 50, 100, 200]
EMA_PERIODS = [12, 26, 50]
RSI_PERIODS = [7, 14, 21]
MOMENTUM_PERIODS = [5, 10, 20]
HIGH_LOW_WINDOWS = [20, 50]
MARKET_SYMBOLS = {"SPY", "QQQ", "^GSPC", "GSPC", "SPX", "IXIC", "^IXIC", "NASDAQ", "VIX", "^VIX"}
MARKET_FEATURE_COLUMNS = ["market_return_1", "market_return_5", "vix_return_1", "vix_return_5", "vix_close"]


def build_feature_frame(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    """Create model features from OHLCV stock data."""
    df = df.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce", utc=True)
    for column in OHLCV_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["Datetime", "stock", *OHLCV_COLUMNS])
    df = df.sort_values(["stock", "Datetime"])

    feature_parts = [
        _build_stock_features(stock_df, include_target)
        for _, stock_df in df.groupby("stock", sort=False)
    ]
    if not feature_parts:
        return pd.DataFrame()

    features = pd.concat(feature_parts, ignore_index=True)
    features = _add_market_features(features)
    features = _add_calendar_features(features)
    features = features.replace([np.inf, -np.inf], np.nan)
    return features


def get_numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"Datetime", "stock", "target_return"}
    return [column for column in df.columns if column not in excluded and pd.api.types.is_numeric_dtype(df[column])]


def _build_stock_features(stock_df: pd.DataFrame, include_target: bool) -> pd.DataFrame:
    stock_df = stock_df.sort_values("Datetime").copy()
    close = stock_df["Close"]
    high = stock_df["High"]
    low = stock_df["Low"]

    for period in LAG_PERIODS:
        stock_df[f"close_lag_{period}"] = close.shift(period)

    for period in RETURN_PERIODS:
        stock_df[f"return_{period}"] = close.pct_change(period)

    for window in ROLLING_WINDOWS:
        rolling_close = close.rolling(window=window, min_periods=window)
        stock_df[f"rolling_mean_{window}"] = rolling_close.mean()
        stock_df[f"rolling_std_{window}"] = rolling_close.std()

    for period in EMA_PERIODS:
        stock_df[f"ema_{period}"] = close.ewm(span=period, adjust=False, min_periods=period).mean()

    for period in RSI_PERIODS:
        stock_df[f"rsi_{period}"] = _rsi(close, period)

    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    stock_df["macd"] = ema_12 - ema_26
    stock_df["macd_signal"] = stock_df["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    stock_df["macd_histogram"] = stock_df["macd"] - stock_df["macd_signal"]

    rolling_20 = close.rolling(window=20, min_periods=20)
    bollinger_middle = rolling_20.mean()
    bollinger_std = rolling_20.std()
    stock_df["bollinger_middle"] = bollinger_middle
    stock_df["bollinger_upper"] = bollinger_middle + 2 * bollinger_std
    stock_df["bollinger_lower"] = bollinger_middle - 2 * bollinger_std
    stock_df["bollinger_bandwidth"] = (stock_df["bollinger_upper"] - stock_df["bollinger_lower"]) / bollinger_middle

    stock_df["atr_14"] = _atr(high, low, close, 14)

    for window in HIGH_LOW_WINDOWS:
        rolling_high = high.rolling(window=window, min_periods=window).max()
        rolling_low = low.rolling(window=window, min_periods=window).min()
        stock_df[f"distance_high_{window}"] = (close - rolling_high) / rolling_high
        stock_df[f"distance_low_{window}"] = (close - rolling_low) / rolling_low

    stock_df["zscore_20"] = (close - bollinger_middle) / bollinger_std

    for period in MOMENTUM_PERIODS:
        stock_df[f"momentum_{period}"] = close - close.shift(period)

    if include_target:
        stock_df["target_return"] = close.pct_change().shift(-1)

    return stock_df


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    dayofweek = df["Datetime"].dt.dayofweek
    month = df["Datetime"].dt.month
    df["dayofweek_sin"] = np.sin(2 * np.pi * dayofweek / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    return df


def _add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    market_mask = df["stock"].isin(MARKET_SYMBOLS)
    market_symbols = {"SPY", "QQQ", "^GSPC", "GSPC", "SPX", "IXIC", "^IXIC", "NASDAQ"}
    market_df = df[market_mask & df["stock"].isin(market_symbols)]
    vix_df = df[market_mask & df["stock"].isin({"VIX", "^VIX"})]

    df = _merge_market_series(df, market_df, "market")
    df = _merge_market_series(df, vix_df, "vix")

    for column in MARKET_FEATURE_COLUMNS:
        if column not in df.columns:
            df[column] = 0.0
    return df


def _merge_market_series(df: pd.DataFrame, market_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if market_df.empty:
        return df

    series = market_df.sort_values("Datetime").drop_duplicates("Datetime", keep="last")
    series = series[["Datetime", "Close"]].copy()
    series[f"{prefix}_return_1"] = series["Close"].pct_change()
    series[f"{prefix}_return_5"] = series["Close"].pct_change(5)
    if prefix == "vix":
        series["vix_close"] = series["Close"]
    series = series.drop(columns=["Close"])

    return pd.merge_asof(
        df.sort_values("Datetime"),
        series.sort_values("Datetime"),
        on="Datetime",
        direction="backward",
    ).sort_values(["stock", "Datetime"])


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
