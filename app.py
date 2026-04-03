import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import joblib
import io
import json
import warnings
warnings.filterwarnings('ignore')

from knn_features import build_features, build_target, calc_ma, to_pandas_resample_rule

# All available feature keys (used for reproducibility config and later for optional feature selection)
FEATURE_KEYS = [
    "rsi_s_z", "rsi_m_z", "rsi_l_z",
    "ma_s_dev_z", "ma_m_dev_z", "ma_l_dev_z",
    "rsi_s_sd_z", "rsi_m_sd_z", "rsi_l_sd_z",
]

def lorentzian_distance(x, y):
    # Sum(log(1 + |x - y|)) — robust-ish alternative to Euclidean
    x = np.asarray(x)
    y = np.asarray(y)
    return float(np.sum(np.log1p(np.abs(x - y))))

st.set_page_config(page_title="Nifty KNN ML Strategy", layout="wide")
st.title("🚀 Nifty KNN Machine Learning Momentum Strategy Dashboard")
st.markdown("**3-Class Classification + Separate prob_up / prob_down** — Better balance & quality")

# ====================== KNN CLASS (SEPARATE PROB_UP + PROB_DOWN) ======================
class KNNMomentumIndicator:
    def __init__(self, k=15, window_size=500, long_threshold=0.68, short_threshold=0.75,
                 momentum_window=5, feat_ma_type="SMA", filter_mode="Price & Fast MA",
                 use_pca=True, p_param=2.0, feature_columns=None, distance_metric="Euclidean"):
        
        self.k = k
        self.window_size = window_size
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.momentum_window = momentum_window
        self.feat_ma_type = feat_ma_type
        self.filter_mode = filter_mode
        self.distance_metric = distance_metric
        # User-controlled: same preprocessing for Euclidean and Lorentzian
        self.use_pca = bool(use_pca)
        self.p_param = p_param
        self.feature_columns = list(feature_columns) if feature_columns is not None else list(FEATURE_KEYS)
        self.pca_n_components = min(3, len(self.feature_columns)) if self.use_pca else 0

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.pca_n_components) if self.use_pca else None
        self.knn = None
        self._lorentzian_X_train = None
        self._lorentzian_y_train = None
        if distance_metric != "Lorentzian":
            # Default to Euclidean (Minkowski p=2)
            self.knn = KNeighborsClassifier(
                n_neighbors=k,
                weights="distance",
                metric="minkowski",
                p=2,
            )
        self.pipeline = Pipeline([
            ("scaler", self.scaler),
            ("pca", self.pca if self.pca is not None else "passthrough"),
            ("knn", self.knn if self.knn is not None else "passthrough"),
        ])

    def _calc_ma(self, src, length):
        return calc_ma(src, length, ma_type=self.feat_ma_type)

    def _feature_engineering(self, df):
        return build_features(df, window_size=self.window_size, feat_ma_type=self.feat_ma_type)

    def fit_on_train_predict_on_test(
        self,
        df: pd.DataFrame,
        train_end_idx: int,
        threshold_ratio: float = 0.72,
    ):
        """
        No look-ahead backtest:
        - Build features using entire df (rolling uses past only).
        - Build target using FUTURE (only for training labels).
        - Fit pipeline on train slice only.
        - Predict probabilities on test slice only (unseen data for the model).
        """
        features = self._feature_engineering(df)
        cols = [c for c in self.feature_columns if c in features.columns]
        if not cols:
            raise ValueError("No selected features found. Choose at least one feature.")
        features = features[cols].copy()
        close = df["Close"].replace([0, np.inf, -np.inf], np.nan).ffill().bfill()
        target = build_target(close, momentum_window=self.momentum_window, threshold_ratio=threshold_ratio)

        # Fit on train
        train_end_idx = int(train_end_idx)
        X_train = features.iloc[:train_end_idx]
        y_train = target.iloc[:train_end_idx]
        X_test = features.iloc[train_end_idx:]

        if self.distance_metric == "Lorentzian":
            # scikit-learn callable metrics are extremely slow; do exact KNN in NumPy.
            self.scaler.fit(X_train)
            Xtr = self.scaler.transform(X_train)
            Xte = self.scaler.transform(X_test)
            if self.pca is not None:
                self.pca.fit(Xtr)
                Xtr = self.pca.transform(Xtr)
                Xte = self.pca.transform(Xte)

            ytr = y_train.to_numpy()
            k = int(self.k)
            prob_up = np.zeros(len(Xte), dtype=float)
            prob_down = np.zeros(len(Xte), dtype=float)
            eps = 1e-9
            for i in range(len(Xte)):
                d = np.log1p(np.abs(Xtr - Xte[i])).sum(axis=1)
                kk = min(k, len(d))
                if kk < len(d):
                    nn_idx = np.argpartition(d, kk - 1)[:kk]
                else:
                    nn_idx = np.arange(len(d))
                nn_d = d[nn_idx]
                w = 1.0 / (nn_d + eps)
                w_sum = float(w.sum()) if len(w) else 0.0
                if w_sum > 0:
                    nn_y = ytr[nn_idx]
                    prob_up[i] = float(w[nn_y == 1].sum()) / w_sum
                    prob_down[i] = float(w[nn_y == -1].sum()) / w_sum
            # For live scoring / joblib: store transformed training set (Lorentzian has no sklearn KNN in pipeline)
            self._lorentzian_X_train = np.asarray(Xtr, dtype=np.float32)
            self._lorentzian_y_train = np.asarray(ytr)
        else:
            self._lorentzian_X_train = None
            self._lorentzian_y_train = None
            self.pipeline.fit(X_train, y_train)
            prob = self.pipeline.predict_proba(X_test)
            classes = list(self.pipeline.classes_)
            prob_up = prob[:, classes.index(1)] if 1 in classes else np.zeros(len(X_test))
            prob_down = prob[:, classes.index(-1)] if -1 in classes else np.zeros(len(X_test))

        prob_up_s = pd.Series(prob_up, index=X_test.index)
        prob_down_s = pd.Series(prob_down, index=X_test.index)
        return prob_up_s, prob_down_s

    def generate_signals(self, df, min_hold_bars=6, profit_target=0.0040, stop_loss=-0.0025):
        df = df.copy()
        raise NotImplementedError("Use generate_signals_no_lookahead() for backtesting.")

    def generate_signals_no_lookahead(
        self,
        df: pd.DataFrame,
        train_end_idx: int,
        min_hold_bars: int = 6,
        profit_target: float = 0.0040,
        stop_loss: float = -0.0025,
        regime_ok: pd.Series = None,
    ) -> pd.DataFrame:
        df = df.copy()
        prob_up, prob_down = self.fit_on_train_predict_on_test(df, train_end_idx=train_end_idx)
        df["prob_up"] = 0.0
        df["prob_down"] = 0.0
        df.loc[prob_up.index, "prob_up"] = prob_up.values
        df.loc[prob_down.index, "prob_down"] = prob_down.values

        # Trend filters (same as before)
        fast_ma = self._calc_ma(df['Close'], 20)
        slow_ma = self._calc_ma(df['Close'], 50)

        filter_bull = pd.Series(True, index=df.index)
        filter_bear = pd.Series(True, index=df.index)

        if self.filter_mode == "Price & Fast MA":
            filter_bull = df['Close'] > fast_ma
            filter_bear = df['Close'] < fast_ma
        elif self.filter_mode == "Fast MA & Slow MA":
            filter_bull = fast_ma > slow_ma
            filter_bear = fast_ma < slow_ma
        elif self.filter_mode == "Price & Fast & Slow":
            filter_bull = (df['Close'] > fast_ma) & (fast_ma > slow_ma)
            filter_bear = (df['Close'] < fast_ma) & (fast_ma < slow_ma)

        df['signal'] = 0
        current_pos = 0
        entry_price = 0.0
        bars_in_trade = 0
        last_dir = 0

        for i in range(len(df)):
            # Don't trade until we're in the TEST region and have enough history
            if i < max(self.window_size + self.momentum_window + 20, int(train_end_idx)):
                continue

            price = df['Close'].iloc[i]
            p_up = df['prob_up'].iloc[i]
            p_down = df['prob_down'].iloc[i]

            raw_long = (p_up > self.long_threshold) and (last_dir <= 0)
            raw_short = (p_down > self.short_threshold) and (last_dir >= 0)

            # === EXIT LOGIC ===
            if current_pos != 0:
                bars_in_trade += 1
                pnl = (price - entry_price) / entry_price

                hit_limit = (bars_in_trade >= min_hold_bars) and (pnl >= profit_target or pnl <= stop_loss)
                strong_opposite = (current_pos == 1 and raw_short) or (current_pos == -1 and raw_long)

                if hit_limit or strong_opposite:
                    df.iloc[i, df.columns.get_loc('signal')] = 0
                    current_pos = 0
                    entry_price = 0.0
                    bars_in_trade = 0
                    last_dir = 0
                    continue

            # === ENTRY LOGIC ===
            ok_regime = regime_ok is None or bool(regime_ok.iloc[i])
            if current_pos == 0:
                if raw_long and filter_bull.iloc[i] and ok_regime:
                    df.iloc[i, df.columns.get_loc("signal")] = 1
                    current_pos = 1
                    entry_price = price
                    bars_in_trade = 0
                    last_dir = 1
                elif raw_short and filter_bear.iloc[i] and ok_regime:
                    df.iloc[i, df.columns.get_loc("signal")] = -1
                    current_pos = -1
                    entry_price = price
                    bars_in_trade = 0
                    last_dir = -1

        df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
        return df

# ====================== RESAMPLE & MAIN (unchanged) ======================
@st.cache_data
def resample_data(df, timeframe):
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    agg = {'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}
    return df.resample(to_pandas_resample_rule(timeframe)).agg(agg).dropna()

# Sidebar
st.sidebar.header("📤 Upload & Settings")
uploaded_file = st.sidebar.file_uploader("Upload Nifty 1-min CSV", type=["csv"])

timeframe = st.sidebar.selectbox(
    "Timeframe",
    ["5min", "15min", "60min", "1D"],
    index=0,
    format_func=lambda x: {"5min": "5 min", "15min": "15 min", "60min": "60 min", "1D": "Daily"}.get(x, x),
    help="Uses pandas offset aliases (minutes = 'min', not deprecated 'T').",
)

bt_period_mode = st.sidebar.radio(
    "Backtest period",
    ["Last N months", "Custom date range"],
    index=0,
    help="Choose how to set the evaluation window. Train is always *before* this window.",
)
if bt_period_mode == "Last N months":
    recent_months = st.sidebar.slider(
        "Backtest recent months",
        3,
        36,
        12,
        help="Evaluation window = last N months. Model trains on data before this.",
    )
    custom_bt_start = custom_bt_end = None
else:
    st.sidebar.caption("Select start and end of backtest window (data range shown after upload).")
    _default_end = date.today()
    _default_start = _default_end - timedelta(days=365)
    custom_bt_start = st.sidebar.date_input("Backtest start date", value=_default_start, key="bt_start")
    custom_bt_end = st.sidebar.date_input("Backtest end date", value=_default_end, key="bt_end")
    recent_months = None

selected_feature_keys = st.sidebar.multiselect(
    "Features to use",
    options=FEATURE_KEYS,
    default=FEATURE_KEYS,
    help="Drop features to exclude from the model. At least one required.",
)

distance_metric = st.sidebar.selectbox(
    "Distance metric",
    ["Euclidean", "Lorentzian"],
    index=0,
    help="Euclidean = Minkowski p=2. Lorentzian = sum(log(1+|Δ|)) on standardized (and optionally PCA) features.",
)

use_pca = st.sidebar.checkbox(
    "Use PCA (after scaling)",
    value=True,
    help="Applies to both distance metrics. Components = min(3, number of selected features). Saved in model bundle for live scoring.",
)

_default_k = st.session_state.get("tune_k", 15)
_default_window = st.session_state.get("tune_window_size", 500)
_default_long = st.session_state.get("tune_long_threshold", 0.65)
_default_short = st.session_state.get("tune_short_threshold", 0.65)
k = st.sidebar.slider("K-Neighbors", 5, 50, _default_k)
window_size = st.sidebar.slider("Learning Window Size", 200, 1500, _default_window)
long_threshold = st.sidebar.number_input("Long Threshold", 0.5, 1.0, _default_long, 0.01)
short_threshold = st.sidebar.number_input("Short Threshold", 0.5, 1.0, _default_short, 0.01)

momentum_window = st.sidebar.slider("Momentum Window", 1, 12, 5)
filter_mode = st.sidebar.selectbox("Filter Mode", ["None","Price & Fast MA","Fast MA & Slow MA","Price & Fast & Slow"], index=1)

min_hold_bars = st.sidebar.slider("Min Hold Bars", 1, 20, 6)
profit_target_pct = st.sidebar.number_input("Profit Target %", 0.01, 5.0, 0.10, 0.01) / 100
stop_loss_pct = st.sidebar.number_input("Stop Loss %", 0.01, 5.0, 0.05, 0.01) / 100

cost_bps = st.sidebar.number_input(
    "Cost per round-trip (bps)",
    0.0,
    100.0,
    0.0,
    1.0,
    help="Brokerage + slippage per round-trip in basis points (1 bps = 0.01%). 0 = no cost.",
)

position_sizing = st.sidebar.radio(
    "Position sizing",
    ["Binary", "Confidence-based"],
    index=0,
    help="Binary = full ±1. Confidence-based = scale size by prob strength in [−1, 1].",
)

regime_filter = st.sidebar.selectbox(
    "Regime filter",
    ["Off", "Volatility", "Trend"],
    index=0,
    help="Only allow entries when regime is favourable. Off = no filter.",
)
regime_vol_max_z = regime_trend_max_pct = None
if regime_filter == "Volatility":
    regime_vol_max_z = st.sidebar.number_input(
        "Max ATR(14) z-score (skip if above)",
        0.5,
        5.0,
        2.5,
        0.1,
        help="Skip new trades when recent ATR z-score is above this (high vol).",
    )
elif regime_filter == "Trend":
    regime_trend_max_pct = st.sidebar.number_input(
        "Max |close − MA50| / MA50 (%) (skip if above)",
        0.5,
        20.0,
        5.0,
        0.5,
        help="Skip new trades when price deviation from MA50 is above this %.",
    )

walk_forward = st.sidebar.selectbox(
    "Walk-forward retrain",
    ["Off", "Every 3 months", "Every 6 months"],
    index=0,
    help="Retrain model every N months over the backtest; stitch segment results.",
)

run_button = st.sidebar.button("🚀 RUN STRATEGY", type="primary")

# Main logic
if uploaded_file is None:
    st.info("Upload your CSV to begin")
    st.stop()

df = pd.read_csv(uploaded_file)
df_resampled = resample_data(df, timeframe)

first_date = df_resampled.index.min()
last_date = df_resampled.index.max()

if bt_period_mode == "Last N months":
    start_date = last_date - pd.DateOffset(months=recent_months)
    end_date = last_date
else:
    # Custom date range
    start_date = pd.Timestamp(custom_bt_start)
    end_date = pd.Timestamp(custom_bt_end) + pd.Timedelta(days=1)  # exclusive end for filtering
    if start_date >= end_date:
        st.error("Backtest start date must be before end date.")
        st.stop()
    # Clip to available data
    start_date = max(start_date, first_date)
    end_date = min(end_date, last_date + pd.Timedelta(days=1))
    if start_date >= end_date:
        st.error("No data in selected range. Choose dates within the file range.")
        st.stop()

train_df = df_resampled[df_resampled.index < start_date].copy()
test_df = df_resampled[(df_resampled.index >= start_date) & (df_resampled.index < end_date)].copy()
backtest_end_show = test_df.index.max() if len(test_df) > 0 else start_date

if len(test_df) == 0 or len(train_df) == 0:
    st.error("Need data both *before* and *inside* the backtest window. Adjust dates or use 'Last N months'.")
    st.stop()

if len(selected_feature_keys) == 0:
    st.error("Select at least one feature to use.")
    st.stop()

st.info(f"Data in file: **{first_date.date()}** – **{last_date.date()}**  |  Backtest window: **{start_date.date()}** – **{backtest_end_show.date()}**  ({len(test_df):,} bars). After RUN, train vs test split will be shown below.")

# ---------- Parameter tuning (grid search on train/validation) ----------
with st.expander("Parameter tuning (timeframe-specific)", expanded=False):
    st.caption("Run a small grid over k, window size, and thresholds on the current *training* data (80%% train / 20%% validation). Best params by validation Sharpe can be applied to the sidebar.")
    run_tune_btn = st.button("Run tuning", key="run_tune")
    if run_tune_btn:
        n_train = len(train_df)
        if n_train < 200:
            st.warning("Not enough training bars for tuning (need at least 200).")
        else:
            val_frac = 0.2
            train_inner_len = int(n_train * (1 - val_frac))
            train_inner = train_df.iloc[:train_inner_len]
            val_df = train_df.iloc[train_inner_len:]
            full_tune = pd.concat([train_inner, val_df])
            tune_train_end = len(train_inner)
            regime_ok_tune = None
            if regime_filter == "Volatility" and regime_vol_max_z is not None:
                atr = (full_tune["High"] - full_tune["Low"]).rolling(14, min_periods=1).mean()
                atr_z = (atr - atr.rolling(50, min_periods=10).mean()) / (atr.rolling(50, min_periods=10).std() + 1e-10)
                regime_ok_tune = atr_z <= regime_vol_max_z
            elif regime_filter == "Trend" and regime_trend_max_pct is not None:
                ma50 = full_tune["Close"].rolling(50, min_periods=1).mean()
                dev_pct = np.abs(full_tune["Close"] - ma50) / (ma50 + 1e-10) * 100
                regime_ok_tune = dev_pct <= regime_trend_max_pct
            grid_k = [10, 15, 20]
            grid_window = [400, 500, 600]
            grid_lt = [0.60, 0.65, 0.70]
            grid_st = [0.60, 0.65, 0.70]
            results_tune = []
            n_combos = len(grid_k) * len(grid_window) * len(grid_lt) * len(grid_st)
            cost_per_trade_tune = (cost_bps / 10000.0) if cost_bps else 0.0
            prog = st.progress(0)
            idx = 0
            for kk in grid_k:
                for ww in grid_window:
                    for lt in grid_lt:
                        for stt in grid_st:
                            idx += 1
                            prog.progress(idx / n_combos)
                            try:
                                m = KNNMomentumIndicator(
                                    k=kk, window_size=ww, long_threshold=lt, short_threshold=stt,
                                    momentum_window=momentum_window, filter_mode=filter_mode,
                                    feature_columns=selected_feature_keys,
                                    distance_metric=distance_metric,
                                    use_pca=use_pca,
                                )
                                res = m.generate_signals_no_lookahead(
                                    full_tune, train_end_idx=tune_train_end,
                                    min_hold_bars=min_hold_bars, profit_target=profit_target_pct, stop_loss=stop_loss_pct,
                                    regime_ok=regime_ok_tune,
                                )
                                res["position"] = res["signal"].replace(0, np.nan).ffill().fillna(0)
                                res["weight"] = res["position"]
                                res["strategy_ret"] = res["weight"] * res["Close"].pct_change()
                                res["prev_position"] = res["position"].shift(1).fillna(0)
                                real_ex = res[(res["position"] == 0) & (res["prev_position"] != 0)]
                                if cost_per_trade_tune > 0 and len(real_ex) > 0:
                                    res.loc[real_ex.index, "strategy_ret"] = res.loc[real_ex.index, "strategy_ret"] - cost_per_trade_tune
                                val_slice = res.iloc[tune_train_end:]
                                if len(val_slice) > 0 and val_slice["strategy_ret"].std() != 0:
                                    sharpe = (val_slice["strategy_ret"].mean() / val_slice["strategy_ret"].std()) * np.sqrt(252 * 4)
                                else:
                                    sharpe = 0.0
                                results_tune.append({"k": kk, "window_size": ww, "long_threshold": lt, "short_threshold": stt, "sharpe": sharpe})
                            except Exception as e:
                                results_tune.append({"k": kk, "window_size": ww, "long_threshold": lt, "short_threshold": stt, "sharpe": np.nan})
            prog.empty()
            if results_tune:
                df_tune = pd.DataFrame(results_tune).sort_values("sharpe", ascending=False)
                st.session_state["tune_results"] = df_tune
    if "tune_results" in st.session_state:
        df_tune = st.session_state["tune_results"]
        st.subheader("Tuning results (validation Sharpe)")
        st.dataframe(df_tune.head(10), use_container_width=True, hide_index=True)
        best = df_tune.iloc[0]
        st.success(f"Best: k={int(best['k'])}, window_size={int(best['window_size'])}, long_threshold={best['long_threshold']:.2f}, short_threshold={best['short_threshold']:.2f} (Sharpe={best['sharpe']:.3f})")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Apply these to strategy", key="apply_tune"):
                st.session_state["tune_k"] = int(best["k"])
                st.session_state["tune_window_size"] = int(best["window_size"])
                st.session_state["tune_long_threshold"] = float(best["long_threshold"])
                st.session_state["tune_short_threshold"] = float(best["short_threshold"])
                st.rerun()
        with col2:
            if st.button("Clear suggested params", key="clear_tune"):
                for key in ["tune_k", "tune_window_size", "tune_long_threshold", "tune_short_threshold", "tune_results"]:
                    st.session_state.pop(key, None)
                st.rerun()

if run_button:
    with st.spinner("Running improved 3-class KNN strategy..."):
        cost_per_trade = (cost_bps / 10000.0) if cost_bps else 0.0
        initial_capital = 100000
        wf_months = 3 if walk_forward == "Every 3 months" else (6 if walk_forward == "Every 6 months" else None)

        if wf_months is None:
            # Single run: train once on all data before backtest, evaluate on backtest window
            model = KNNMomentumIndicator(
                k=k,
                window_size=window_size,
                long_threshold=long_threshold,
                short_threshold=short_threshold,
                momentum_window=momentum_window,
                filter_mode=filter_mode,
                feature_columns=selected_feature_keys,
                distance_metric=distance_metric,
                use_pca=use_pca,
            )
            full_df = pd.concat([train_df, test_df])
            train_end_idx = len(train_df)
            regime_ok = None
            if regime_filter == "Volatility" and regime_vol_max_z is not None:
                atr = (full_df["High"] - full_df["Low"]).rolling(14, min_periods=1).mean()
                atr_z = (atr - atr.rolling(50, min_periods=10).mean()) / (atr.rolling(50, min_periods=10).std() + 1e-10)
                regime_ok = atr_z <= regime_vol_max_z
            elif regime_filter == "Trend" and regime_trend_max_pct is not None:
                ma50 = full_df["Close"].rolling(50, min_periods=1).mean()
                dev_pct = np.abs(full_df["Close"] - ma50) / (ma50 + 1e-10) * 100
                regime_ok = dev_pct <= regime_trend_max_pct
            result = model.generate_signals_no_lookahead(
                full_df, train_end_idx=train_end_idx,
                min_hold_bars=min_hold_bars, profit_target=profit_target_pct, stop_loss=stop_loss_pct,
                regime_ok=regime_ok,
            )
            result["position"] = result["signal"].replace(0, np.nan).ffill().fillna(0)
            if position_sizing == "Binary":
                result["weight"] = result["position"]
            else:
                w = np.zeros(len(result))
                pos = result["position"].values
                p_up = result["prob_up"].values
                p_dn = result["prob_down"].values
                lt, short_thr = model.long_threshold, model.short_threshold
                for i in range(len(result)):
                    if pos[i] == 1:
                        w[i] = np.clip((p_up[i] - lt) / (1 - lt + 1e-10), 0, 1)
                    elif pos[i] == -1:
                        w[i] = -np.clip((p_dn[i] - short_thr) / (1 - short_thr + 1e-10), 0, 1)
                result["weight"] = w
            result["strategy_ret"] = result["weight"] * result["Close"].pct_change()
            result["prev_position"] = result["position"].shift(1).fillna(0)
            real_exits = result[(result["position"] == 0) & (result["prev_position"] != 0)].copy()
            if cost_per_trade > 0 and len(real_exits) > 0:
                result.loc[real_exits.index, "strategy_ret"] = result.loc[real_exits.index, "strategy_ret"] - cost_per_trade
            result["cum_strategy"] = (1 + result["strategy_ret"].fillna(0)).cumprod()
            result["cum_buyhold"] = (1 + result["Close"].pct_change().fillna(0)).cumprod()
            result['exit_pnl'] = 0.0
            entry_price_series = pd.Series(np.nan, index=result.index)
            current_entry = 0.0
            for i in range(len(result)):
                if result['signal'].iloc[i] != 0 and result['position'].iloc[i] != result['prev_position'].iloc[i]:
                    current_entry = result['Close'].iloc[i]
                entry_price_series.iloc[i] = current_entry
            if len(real_exits) > 0:
                exit_idx = real_exits.index
                result.loc[exit_idx, 'exit_pnl'] = (result['Close'].reindex(exit_idx) - entry_price_series.reindex(exit_idx)) / entry_price_series.reindex(exit_idx)
            result['exit_pnl'] = result['exit_pnl'].fillna(0)
            result['equity_inr'] = initial_capital * result['cum_strategy']
            result_display = result.iloc[train_end_idx:].copy()
            if len(result_display) > 0:
                result_display['cum_strategy'] = result_display['cum_strategy'] / result_display['cum_strategy'].iloc[0]
                result_display['cum_buyhold'] = result_display['cum_buyhold'] / result_display['cum_buyhold'].iloc[0]
                result_display['equity_inr'] = initial_capital * result_display['cum_strategy']
        else:
            # Walk-forward: retrain every wf_months over the backtest, stitch segment results
            freq = f"{wf_months}M"
            segments = [g for _, g in test_df.groupby(pd.Grouper(freq=freq)) if len(g) > 0]
            if not segments:
                st.error("Backtest window too short for walk-forward segments. Widen the backtest period.")
                st.stop()
            train_so_far = train_df
            result_segments = []
            model = None
            for seg_df in segments:
                full_wf = pd.concat([train_so_far, seg_df])
                train_end_idx_wf = len(train_so_far)
                regime_ok_wf = None
                if regime_filter == "Volatility" and regime_vol_max_z is not None:
                    atr = (full_wf["High"] - full_wf["Low"]).rolling(14, min_periods=1).mean()
                    atr_z = (atr - atr.rolling(50, min_periods=10).mean()) / (atr.rolling(50, min_periods=10).std() + 1e-10)
                    regime_ok_wf = atr_z <= regime_vol_max_z
                elif regime_filter == "Trend" and regime_trend_max_pct is not None:
                    ma50 = full_wf["Close"].rolling(50, min_periods=1).mean()
                    dev_pct = np.abs(full_wf["Close"] - ma50) / (ma50 + 1e-10) * 100
                    regime_ok_wf = dev_pct <= regime_trend_max_pct
                model = KNNMomentumIndicator(
                    k=k, window_size=window_size, long_threshold=long_threshold, short_threshold=short_threshold,
                    momentum_window=momentum_window, filter_mode=filter_mode, feature_columns=selected_feature_keys,
                    distance_metric=distance_metric,
                    use_pca=use_pca,
                )
                res = model.generate_signals_no_lookahead(
                    full_wf, train_end_idx=train_end_idx_wf,
                    min_hold_bars=min_hold_bars, profit_target=profit_target_pct, stop_loss=stop_loss_pct,
                    regime_ok=regime_ok_wf,
                )
                res["position"] = res["signal"].replace(0, np.nan).ffill().fillna(0)
                if position_sizing == "Binary":
                    res["weight"] = res["position"]
                else:
                    w = np.zeros(len(res))
                    pos = res["position"].values
                    p_up = res["prob_up"].values
                    p_dn = res["prob_down"].values
                    lt, short_thr = model.long_threshold, model.short_threshold
                    for i in range(len(res)):
                        if pos[i] == 1:
                            w[i] = np.clip((p_up[i] - lt) / (1 - lt + 1e-10), 0, 1)
                        elif pos[i] == -1:
                            w[i] = -np.clip((p_dn[i] - short_thr) / (1 - short_thr + 1e-10), 0, 1)
                    res["weight"] = w
                res["strategy_ret"] = res["weight"] * res["Close"].pct_change()
                res["prev_position"] = res["position"].shift(1).fillna(0)
                real_ex_wf = res[(res["position"] == 0) & (res["prev_position"] != 0)]
                if cost_per_trade > 0 and len(real_ex_wf) > 0:
                    res.loc[real_ex_wf.index, "strategy_ret"] = res.loc[real_ex_wf.index, "strategy_ret"] - cost_per_trade
                res["cum_strategy"] = (1 + res["strategy_ret"].fillna(0)).cumprod()
                res["cum_buyhold"] = (1 + res["Close"].pct_change().fillna(0)).cumprod()
                res["exit_pnl"] = 0.0
                entry_ser = pd.Series(np.nan, index=res.index)
                cur_entry = 0.0
                for i in range(len(res)):
                    if res["signal"].iloc[i] != 0 and res["position"].iloc[i] != res["prev_position"].iloc[i]:
                        cur_entry = res["Close"].iloc[i]
                    entry_ser.iloc[i] = cur_entry
                if len(real_ex_wf) > 0:
                    exit_idx = real_ex_wf.index
                    res.loc[exit_idx, "exit_pnl"] = (res["Close"].reindex(exit_idx) - entry_ser.reindex(exit_idx)) / entry_ser.reindex(exit_idx)
                res["exit_pnl"] = res["exit_pnl"].fillna(0)
                seg_slice = res.iloc[train_end_idx_wf:].copy()
                result_segments.append(seg_slice)
                train_so_far = pd.concat([train_so_far, seg_df])
            result_display = pd.concat(result_segments).sort_index()
            result_display["cum_strategy"] = (1 + result_display["strategy_ret"].fillna(0)).cumprod()
            result_display["cum_strategy"] = result_display["cum_strategy"] / result_display["cum_strategy"].iloc[0]
            result_display["cum_buyhold"] = (1 + result_display["Close"].pct_change().fillna(0)).cumprod()
            result_display["cum_buyhold"] = result_display["cum_buyhold"] / result_display["cum_buyhold"].iloc[0]
            result_display["equity_inr"] = initial_capital * result_display["cum_strategy"]
            train_end_idx = len(train_df)  # for run_config / message

        if len(result_display) > 0:
            total_ret = result_display['cum_strategy'].iloc[-1] - 1
            bh_ret = result_display['cum_buyhold'].iloc[-1] - 1
            sharpe = (result_display['strategy_ret'].mean() / result_display['strategy_ret'].std() * np.sqrt(252*4)) if result_display['strategy_ret'].std() != 0 else 0
            trades = (result_display['signal'].diff() != 0).sum() // 2
            real_exits = result_display[(result_display['position'] == 0) & (result_display['prev_position'] != 0)]
            target_hits = (result_display['exit_pnl'] >= profit_target_pct).sum()
            sl_hits = (result_display['exit_pnl'] <= stop_loss_pct).sum()
            other_exits = len(real_exits) - target_hits - sl_hits
            long_trades = (result_display['signal'] == 1).sum()
            short_trades = (result_display['signal'] == -1).sum()
            total_dir = long_trades + short_trades
            long_pct = long_trades / total_dir * 100 if total_dir > 0 else 0
            short_pct = short_trades / total_dir * 100 if total_dir > 0 else 0
            long_returns = result_display['strategy_ret'][(result_display['position'] > 0)]
            short_returns = result_display['strategy_ret'][(result_display['position'] < 0)]
            long_win_rate = (long_returns > 0).mean() * 100 if len(long_returns) > 0 else 0
            short_win_rate = (short_returns > 0).mean() * 100 if len(short_returns) > 0 else 0
            cum_ret = result_display['cum_strategy'].fillna(1)
            rolling_max = cum_ret.cummax()
            drawdown = (cum_ret - rolling_max) / rolling_max
            max_dd = drawdown.min() * 100 if not drawdown.empty else 0
            final_equity = result_display['equity_inr'].iloc[-1]
            profit_inr = final_equity - initial_capital
            capital_return_pct = (final_equity - initial_capital) / initial_capital * 100
        else:
            result_display = result  # fallback when no test bars
            total_ret = result['cum_strategy'].iloc[-1] - 1
            bh_ret = result['cum_buyhold'].iloc[-1] - 1
            sharpe = (result['strategy_ret'].mean() / result['strategy_ret'].std() * np.sqrt(252*4)) if result['strategy_ret'].std() != 0 else 0
            trades = (result['signal'].diff() != 0).sum() // 2
            real_exits = result[(result['position'] == 0) & (result['prev_position'] != 0)]
            target_hits = (result['exit_pnl'] >= profit_target_pct).sum()
            sl_hits = (result['exit_pnl'] <= stop_loss_pct).sum()
            other_exits = len(real_exits) - target_hits - sl_hits
            long_trades = (result['signal'] == 1).sum()
            short_trades = (result['signal'] == -1).sum()
            total_dir = long_trades + short_trades
            long_pct = long_trades / total_dir * 100 if total_dir > 0 else 0
            short_pct = short_trades / total_dir * 100 if total_dir > 0 else 0
            long_returns = result['strategy_ret'][(result['position'] > 0)]
            short_returns = result['strategy_ret'][(result['position'] < 0)]
            long_win_rate = (long_returns > 0).mean() * 100 if len(long_returns) > 0 else 0
            short_win_rate = (short_returns > 0).mean() * 100 if len(short_returns) > 0 else 0
            cum_ret = result['cum_strategy'].fillna(1)
            max_dd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min() * 100 if not result.empty else 0

        train_start_ts = train_df.index[0]
        train_end_ts = train_df.index[-1]
        test_start_ts = test_df.index[0]
        test_end_ts = test_df.index[-1]

        # Build trade list (entry/exit details) for download — use result_display so only test-period trades
        result_display['prev_position'] = result_display['position'].shift(1).fillna(0)
        entry_bars = result_display[(result_display['signal'] != 0) & (result_display['position'] != result_display['prev_position'])]
        exit_bars = result_display[(result_display['position'] == 0) & (result_display['prev_position'] != 0)]
        n_trades = min(len(entry_bars), len(exit_bars))
        trades_list = []
        for t in range(n_trades):
            entry_row = entry_bars.iloc[t]
            exit_row = exit_bars.iloc[t]
            entry_time = entry_row.name
            exit_time = exit_row.name
            entry_price = entry_row['Close']
            exit_price = exit_row['Close']
            direction = 'Long' if entry_row['signal'] == 1 else 'Short'
            bars_held = (result_display.index.get_loc(exit_time) - result_display.index.get_loc(entry_time)) if entry_time in result_display.index and exit_time in result_display.index else np.nan
            return_pct = exit_row["exit_pnl"] - cost_per_trade  # net after cost
            equity_at_entry = result_display["equity_inr"].loc[entry_time]
            return_inr = equity_at_entry * return_pct
            if return_pct >= profit_target_pct:
                exit_reason = 'Profit Target'
            elif return_pct <= stop_loss_pct:
                exit_reason = 'Stop Loss'
            else:
                exit_reason = 'Signal'
            trades_list.append({
                'Trade_No': t + 1,
                'Entry_Time': entry_time,
                'Exit_Time': exit_time,
                'Direction': direction,
                'Entry_Price': round(entry_price, 2),
                'Exit_Price': round(exit_price, 2),
                'Bars_Held': int(bars_held) if not np.isnan(bars_held) else '',
                'Return_Pct': round(return_pct * 100, 4),
                'Return_INR': round(return_inr, 2),
                'Equity_At_Entry_INR': round(equity_at_entry, 2),
                'Exit_Reason': exit_reason,
            })
        trades_df = pd.DataFrame(trades_list) if trades_list else pd.DataFrame()

        # ==================== DISPLAY ====================
        wf_note = f" Walk-forward: retrain every {wf_months} months ({len(segments) if wf_months else 0} segments)." if wf_months else ""
        st.success(f"**Train:** {train_start_ts} → {train_end_ts} ({train_end_idx:,} bars)  |  **Test (backtest):** {test_start_ts} → {test_end_ts} ({len(result_display):,} bars). Metrics & charts are for the test period only (no look-ahead).{wf_note}")
        st.caption("Returns are often lower than before because the model is tested on *unseen* data (realistic). Lower **Train %** = longer test period but less training data.")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Strategy Return", f"{total_ret*100:.2f}%", f"{capital_return_pct:.2f}%")
        col2.metric("Return (INR)", f"₹{profit_inr:,.0f}", f"₹{initial_capital:,} → ₹{final_equity:,.0f}")
        col3.metric("Buy & Hold", f"{bh_ret*100:.2f}%")
        col4.metric("Sharpe Ratio", f"{sharpe:.2f}")
        col5.metric("Max Drawdown", f"{max_dd:.2f}%", delta_color="inverse")

        st.markdown("---")

        colA, colB, colC, colD = st.columns(4)
        colA.metric("Total Trades", trades)
        colB.metric("Long / Short", f"{long_trades} / {short_trades}", f"{long_pct:.0f}% – {short_pct:.0f}%")
        colC.metric("Win Rate (L/S)", f"{long_win_rate:.1f}% / {short_win_rate:.1f}%")
        colD.metric("Target vs SL hit", f"{target_hits} vs {sl_hits}", f"Other: {other_exits}")

        st.caption(f"Simulated on ₹{initial_capital:,} starting capital → final ₹{final_equity:,.0f}")


        # ==================== CHARTS ====================
        tab1, tab2 = st.tabs(["Price + Signals", "Equity Curve"])

        fig1 = make_subplots(rows=1, cols=1)
        fig1.add_trace(go.Scatter(x=result_display.index, y=result_display['Close'], name="Close", line=dict(color="#1E90FF")))
        
        buys  = result_display[result_display['signal'] == 1]
        sells = result_display[result_display['signal'] == -1]

        fig1.add_trace(go.Scatter(x=buys.index, y=buys['Close'], mode='markers',
                                  marker=dict(symbol='triangle-up', size=14, color='lime', line=dict(color='darkgreen', width=1.5)),
                                  name="BUY"))
        fig1.add_trace(go.Scatter(x=sells.index, y=sells['Close'], mode='markers',
                                  marker=dict(symbol='triangle-down', size=14, color='red', line=dict(color='darkred', width=1.5)),
                                  name="SELL"))

        real_exits_display = result_display[(result_display['position'] == 0) & (result_display['prev_position'] != 0)]
        fig1.add_trace(go.Scatter(
            x=real_exits_display.index, y=real_exits_display['Close'], mode='markers',
            marker=dict(symbol='x', size=10, color='white', line=dict(color='black', width=1.5)),
            name="EXIT"
        ))

        fig1.update_layout(title=f"Nifty {timeframe} – KNN Classification Momentum", height=550)
        fig1.update_xaxes(rangebreaks=[
            dict(bounds=[15.5, 9.25], pattern="hour"),
            dict(bounds=["sat", "mon"])
        ])

        tab1.plotly_chart(fig1, width='stretch')

        # Equity curve (INR — test period only)
        buyhold_inr = initial_capital * result_display['cum_buyhold']
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=result_display.index, y=result_display['equity_inr'], name="Strategy (₹)", line=dict(color="green", width=2.5)))
        fig2.add_trace(go.Scatter(x=result_display.index, y=buyhold_inr, name="Buy & Hold (₹)", line=dict(color="gray")))
        fig2.update_layout(title="Cumulative Performance (INR on ₹1L)", height=400, yaxis_tickprefix="₹")
        fig2.update_xaxes(rangebreaks=[
            dict(bounds=[15.5, 9.25], pattern="hour"),
            dict(bounds=["sat", "mon"])
        ])
        tab2.plotly_chart(fig2, width='stretch')

        # Reproducibility: full run config (serializable for JSON and joblib)
        run_config = {
            "symbol": "Nifty",
            "timeframe": timeframe,
            "backtest_period_mode": bt_period_mode,
            "train_start_ts": str(train_start_ts),
            "train_end_ts": str(train_end_ts),
            "test_start_ts": str(test_start_ts),
            "test_end_ts": str(test_end_ts),
            "train_bars": len(train_df),
            "test_bars": len(test_df),
            "params": {
                "k": model.k,
                "window_size": model.window_size,
                "long_threshold": model.long_threshold,
                "short_threshold": model.short_threshold,
                "momentum_window": model.momentum_window,
                "feat_ma_type": model.feat_ma_type,
                "filter_mode": model.filter_mode,
                "use_pca": model.use_pca,
                "pca_n_components": model.pca_n_components,
                "p_param": model.p_param,
                "distance_metric": distance_metric,
                "min_hold_bars": min_hold_bars,
                "profit_target_pct": profit_target_pct,
                "stop_loss_pct": stop_loss_pct,
                "cost_bps": cost_bps,
            },
            "position_sizing": position_sizing,
            "regime_filter": regime_filter,
            "regime_vol_max_z": regime_vol_max_z,
            "regime_trend_max_pct": regime_trend_max_pct,
            "features_used": list(selected_feature_keys),
            "walk_forward_months": wf_months,
        }
        if "tune_results" in st.session_state:
            tr = st.session_state["tune_results"]
            best_row = tr.iloc[0]
            run_config["tuning"] = {
                "best_params": {"k": int(best_row["k"]), "window_size": int(best_row["window_size"]), "long_threshold": float(best_row["long_threshold"]), "short_threshold": float(best_row["short_threshold"]), "validation_sharpe": float(best_row["sharpe"])},
                "grid": {"k": [10, 15, 20], "window_size": [400, 500, 600], "long_threshold": [0.60, 0.65, 0.70], "short_threshold": [0.60, 0.65, 0.70]},
            }

        # Persist downloads so buttons remain after Streamlit reruns
        _model_buf = io.BytesIO()
        lorentzian_cache = None
        if getattr(model, "distance_metric", "") == "Lorentzian" and getattr(model, "_lorentzian_X_train", None) is not None:
            lorentzian_cache = {
                "X_train": model._lorentzian_X_train,
                "y_train": model._lorentzian_y_train,
                "k": int(model.k),
            }
        joblib.dump({
            "pipeline": model.pipeline,
            "params": run_config["params"],
            "timeframe": timeframe,
            "run_config": run_config,
            "lorentzian_cache": lorentzian_cache,
        }, _model_buf)
        _model_buf.seek(0)

        config_json_bytes = json.dumps(run_config, indent=2).encode()
        st.session_state["downloads"] = {
            "timeframe": timeframe,
            "results_csv_bytes": result_display.to_csv().encode(),
            "trades_csv_bytes": (trades_df.to_csv(index=False).encode() if not trades_df.empty else None),
            "model_joblib_bytes": _model_buf.getvalue(),
            "config_json_bytes": config_json_bytes,
        }

        with st.expander("How to deploy this strategy in the real market"):
            st.markdown("""
            **Does the strategy get saved as a model that can be called on live prices?**  
            Yes. Saving includes the **fitted pipeline** (StandardScaler → optional PCA → KNN *or* passthrough for Lorentzian), plus **`params`** and **`run_config`** (`use_pca`, `distance_metric`, `features_used`, etc.). Lorentzian models also include **`lorentzian_cache`** (transformed training matrix for exact KNN on live).

            **What to save**
            - The **fitted pipeline** (`model.pipeline`): StandardScaler, optional PCA (`use_pca` in params), and for Euclidean a fitted `KNeighborsClassifier`.
            - **Parameters** and **`run_config`**: `window_size`, `momentum_window`, `long_threshold`, `short_threshold`, `filter_mode`, `feat_ma_type`, `use_pca`, `pca_n_components`, `distance_metric`, `features_used`, etc.

            **How to run on live market**
            1. **Data:** Maintain a rolling buffer of OHLC bars at the **same timeframe** as backtest (e.g. 5min). Feed new candles from your broker (e.g. Fyers) as they close.
            2. **Features:** From the buffer, compute the same 9 features (RSI z-scores, MA deviation z-scores, etc.) using the same `window_size` and formulas. Use only the **last row** (current bar) for prediction.
            3. **Predict:** Use **`score_latest_candle.py`**: it applies the saved `use_pca` and `distance_metric` (Euclidean → `predict_proba` on raw feature row; Lorentzian → scale (+PCA) then weighted neighbors using `lorentzian_cache`). You get `prob_up` and `prob_down`.
            4. **Signal:** Apply your thresholds (`long_threshold`, `short_threshold`) and trend filter (`filter_mode`). Emit Long / Short / Flat and optionally send orders via your broker API.

            **Important**
            - The model was trained on **past data**. For live use you can either (a) use the same saved model and refit periodically (e.g. daily/weekly) on recent history, or (b) retrain on a rolling window before each prediction (slower).
            - Use the **same timeframe and symbol** (e.g. Nifty 5-min) as in backtest. Tick-by-tick you would aggregate ticks into that timeframe first, then run the model on closed bars.
            - See **DEPLOY.md** in the project for a step-by-step deployment guide and example code to load the saved model and run on live bars.
            """)

if "downloads" in st.session_state:
    st.markdown("---")
    _dl = st.session_state["downloads"]
    _tf = _dl.get("timeframe", "NA")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.download_button(
            "Download Results CSV",
            _dl["results_csv_bytes"],
            f"knn_classification_{_tf}.csv",
            "text/csv",
            key="dl_results_persist",
        )
    with d2:
        if _dl.get("trades_csv_bytes"):
            st.download_button(
                "Download Entry/Exit Trade Details",
                _dl["trades_csv_bytes"],
                f"knn_trade_details_{_tf}.csv",
                "text/csv",
                key="dl_trades_persist",
            )
        else:
            st.caption("No closed trades for trade-details download.")
    with d3:
        st.download_button(
            "Save model (for live)",
            _dl["model_joblib_bytes"],
            f"knn_model_{_tf}.joblib",
            "application/octet-stream",
            key="dl_model_persist",
        )
    with d4:
        st.download_button(
            "Download run config (JSON)",
            _dl["config_json_bytes"],
            f"run_config_{_tf}.json",
            "application/json",
            key="dl_config_persist",
        )

st.caption("Classification + Neutral + Gaussian weighting logic")
