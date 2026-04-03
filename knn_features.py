import numpy as np
import pandas as pd


def calc_ma(src: pd.Series, length: int, ma_type: str = "SMA") -> pd.Series:
    len_ = max(1, int(length))
    if ma_type == "EMA":
        return src.ewm(span=len_, adjust=False).mean()
    return src.rolling(len_, min_periods=1).mean()


def calc_rsi(src: pd.Series, period: int) -> pd.Series:
    period = max(1, int(period))
    delta = src.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=1).mean()
    loss = -delta.clip(upper=0).rolling(period, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    return 100 - 100 / (1 + rs)


def zscore_rolling(s: pd.Series, window: int, min_periods: int = 15) -> pd.Series:
    r = s.rolling(int(window), min_periods=min_periods)
    return (s - r.mean()) / (r.std() + 1e-10)


def build_features(
    df: pd.DataFrame,
    window_size: int,
    feat_ma_type: str = "SMA",
    rsi_periods: tuple[int, int, int] = (2, 3, 5),
    ma_periods: tuple[int, int, int] = (2, 3, 5),
    signal_len: int = 4,
) -> pd.DataFrame:
    """
    Feature engineering used by both backtest and live scoring.
    Returns a DataFrame indexed like df with 9 feature columns.
    """
    close = df["Close"].replace([0, np.inf, -np.inf], np.nan).ffill().bfill()

    rsi_s = calc_rsi(close, rsi_periods[0])
    rsi_m = calc_rsi(close, rsi_periods[1])
    rsi_l = calc_rsi(close, rsi_periods[2])

    ma_s = calc_ma(close, ma_periods[0], ma_type=feat_ma_type)
    ma_m = calc_ma(close, ma_periods[1], ma_type=feat_ma_type)
    ma_l = calc_ma(close, ma_periods[2], ma_type=feat_ma_type)

    ma_s_dev = (close - ma_s) / (ma_s + 1e-10) * 100
    ma_m_dev = (close - ma_m) / (ma_m + 1e-10) * 100
    ma_l_dev = (close - ma_l) / (ma_l + 1e-10) * 100

    rsi_s_sig = rsi_s - rsi_s.rolling(int(signal_len), min_periods=1).mean()
    rsi_m_sig = rsi_m - rsi_m.rolling(int(signal_len), min_periods=1).mean()
    rsi_l_sig = rsi_l - rsi_l.rolling(int(signal_len), min_periods=1).mean()

    features = pd.DataFrame(
        {
            "rsi_s_z": zscore_rolling(rsi_s, window_size, min_periods=15),
            "rsi_m_z": zscore_rolling(rsi_m, window_size, min_periods=15),
            "rsi_l_z": zscore_rolling(rsi_l, window_size, min_periods=15),
            "ma_s_dev_z": zscore_rolling(ma_s_dev, window_size, min_periods=15),
            "ma_m_dev_z": zscore_rolling(ma_m_dev, window_size, min_periods=15),
            "ma_l_dev_z": zscore_rolling(ma_l_dev, window_size, min_periods=15),
            "rsi_s_sd_z": zscore_rolling(rsi_s_sig, window_size, min_periods=15),
            "rsi_m_sd_z": zscore_rolling(rsi_m_sig, window_size, min_periods=15),
            "rsi_l_sd_z": zscore_rolling(rsi_l_sig, window_size, min_periods=15),
        },
        index=df.index,
    ).fillna(0)

    return features


def build_target(
    close: pd.Series,
    momentum_window: int,
    threshold_ratio: float = 0.72,
) -> pd.Series:
    """
    3-class target:
      +1 if next momentum_window closes are mostly above current (>= threshold_ratio)
      -1 if mostly below (>= threshold_ratio)
       0 otherwise

    Note: This is for TRAINING ONLY. For TEST/live you will not use future data.
    """
    mw = int(momentum_window)
    target = pd.Series(0, index=close.index, dtype=int)
    for i in range(len(close) - mw):
        future = close.iloc[i + 1 : i + 1 + mw]
        current = close.iloc[i]
        up_ratio = (future > current).mean()
        down_ratio = (future < current).mean()
        if up_ratio >= threshold_ratio:
            target.iloc[i] = 1
        elif down_ratio >= threshold_ratio:
            target.iloc[i] = -1
    return target

