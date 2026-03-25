import math
import smtplib
import time
from dataclasses import dataclass
from datetime import datetime
from email.mime.text import MIMEText
from io import StringIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

st.set_page_config(
    page_title="Trading Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Theme System ─────────────────────────────────────────────────────────────
_THEMES = {
    "TradingView Dark": {
        "bg": "#131722", "card": "#1e222d", "border": "#2a2e39",
        "text": "#d1d4dc", "sub": "#787b86", "accent": "#2962ff",
        "chart": "#131722", "grid": "#2a2e39",
    },
    "TradingView Light": {
        "bg": "#ffffff", "card": "#f0f3fa", "border": "#e0e3eb",
        "text": "#131722", "sub": "#787b86", "accent": "#2962ff",
        "chart": "#ffffff", "grid": "#e0e3eb",
    },
    "Dark": {
        "bg": "#0d1117", "card": "#161b22", "border": "#21262d",
        "text": "#e6edf3", "sub": "#8b949e", "accent": "#3b82f6",
        "chart": "#0d1117", "grid": "#21262d",
    },
    "Midnight Blue": {
        "bg": "#07090f", "card": "#0d1121", "border": "#1a2035",
        "text": "#e2e8f0", "sub": "#64748b", "accent": "#6366f1",
        "chart": "#07090f", "grid": "#1a2035",
    },
    "Bloomberg Orange": {
        "bg": "#0a0a0a", "card": "#111111", "border": "#2a1e00",
        "text": "#ff8c00", "sub": "#996600", "accent": "#ff8c00",
        "chart": "#080800", "grid": "#1f1800",
    },
    "Cyberpunk": {
        "bg": "#09090b", "card": "#18181b", "border": "#27272a",
        "text": "#f4f4f5", "sub": "#a1a1aa", "accent": "#facc15",
        "chart": "#09090b", "grid": "#27272a",
    },
    "Hacker Terminal": {
        "bg": "#000000", "card": "#001100", "border": "#003300",
        "text": "#00ff00", "sub": "#00aa00", "accent": "#00ff00",
        "chart": "#000000", "grid": "#002200",
    }
}

def get_theme_colors():
    return _THEMES.get(st.session_state.get("theme", "TradingView Dark"), _THEMES["TradingView Dark"])

def get_theme_font():
    f = st.session_state.get("font", "System UI")
    if f == "Monospace": return "'Courier New', monospace"
    if f == "Sans-serif": return "'Inter', 'Helvetica Neue', sans-serif"
    if f == "Serif": return "'Georgia', 'Times New Roman', serif"
    return "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

def inject_theme():
    c = get_theme_colors()
    ff = get_theme_font()
    st.markdown(f"""<style>
    .stApp {{background-color:{c['bg']} !important;}}
    body, .stMarkdown, .stText, label, .stSelectbox label, .stSlider label {{
        font-family:{ff} !important; color:{c['sub']} !important;
    }}
    h1, h2, h3, h4, h5, h6 {{ color:{c['text']} !important; font-family:{ff} !important; }}
    [data-testid="stMetric"] {{background-color:{c['card']};border:1px solid {c['border']};border-radius:6px;padding:12px 16px;}}
    [data-testid="stMetricLabel"] {{color:{c['sub']} !important; font-size: 0.75rem; letter-spacing: 0.05em;}}
    [data-testid="stMetricValue"] {{color:{c['text']} !important; font-weight: 700;}}
    .stTabs [data-baseweb="tab-list"] {{background-color:{c['card']}; border-radius: 6px; padding: 4px;}}
    .stTabs [data-baseweb="tab"] {{color:{c['sub']}; font-family:{ff}; font-size: 0.85rem; font-weight:600; padding: 6px 16px;}}
    .stTabs [aria-selected="true"] {{color:{c['accent']} !important; border-bottom: 2px solid {c['accent']}; background: transparent !important;}}
    .verdict-card {{padding: 12px 20px; border-radius: 6px; font-size: 1.3rem; font-weight: 700; font-family: {ff}; text-align: center; margin-bottom: 12px; letter-spacing: 0.02em;}}
    .verdict-buy  {{background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; color: #10b981;}}
    .verdict-sell {{background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; color: #ef4444;}}
    .verdict-neutral {{background: rgba(120, 113, 108, 0.1); border: 1px solid #78716c; color: {c['sub']};}}
    .risk-card, .reason-box, .pick-card {{background-color:{c['card']};border:1px solid {c['border']};border-radius:6px;padding:12px 16px;}}
    .reason-box {{border-left: 3px solid {c['accent']}; color: {c['text']}; font-size: 0.85rem; margin: 6px 0;}}
    .risk-row {{display: flex; justify-content: space-between; align-items: center; padding: 5px 0; border-bottom: 1px solid {c['border']};}}
    .risk-row:last-child {{border-bottom: none;}}
    .risk-label {{color: {c['sub']}; font-size: 0.8rem;}}
    div[data-baseweb="select"]>div, div[data-baseweb="input"]>div {{background-color:{c['card']} !important;border-color:{c['border']} !important; color: {c['text']} !important;}}
    .stButton>button {{background-color:{c['accent']} !important; color: #ffffff !important; border-radius: 6px !important; font-weight: 600 !important; border:none;}}
    .streamlit-expanderHeader {{background-color:{c['card']} !important; border-radius: 6px !important; font-family:{ff}; color:{c['text']} !important;}}
    hr {{border-color:{c['border']};}}
    .pick-ticker {{ color: {c['accent']}; font-size: 1.1rem; font-weight: 700; }}
    .pick-score  {{ color: #10b981; font-size: 1.5rem; font-weight: 700; }}
    </style>""", unsafe_allow_html=True)


# ============================================================
# Data / universe helpers
# ============================================================
@st.cache_data(ttl=60 * 60)
def fetch_universe() -> List[str]:
    fallback = sorted(list(dict.fromkeys([
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "COST", "AMD", 
        "NFLX", "ADBE", "QCOM", "CSCO", "INTC", "AMGN", "TXN", "PEP", "CMCSA", "TMUS",
    ])))
    headers = {"User-Agent": "Mozilla/5.0 (compatible; TradingTerminal/3.0)"}

    def _tables(url):
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        return pd.read_html(StringIO(r.text))

    try:
        sp = _tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]["Symbol"].astype(str).tolist()
        ndx_tables = _tables("https://en.wikipedia.org/wiki/Nasdaq-100")
        ndx = []
        for t in ndx_tables:
            cols = [str(c).strip().lower() for c in t.columns]
            if "ticker" in cols:
                ndx = t[t.columns[cols.index("ticker")]].astype(str).tolist()
                break
        if not ndx: raise ValueError("no ndx")
        cleaned = [s.replace(".", "-").strip().upper() for s in sp + ndx if s]
        return sorted(list(dict.fromkeys(cleaned)))
    except Exception:
        return fallback

DOW30 = sorted([
    "AAPL","AMGN","AXP","BA","CAT","CRM","CSCO","CVX","DIS","DOW",
    "GS","HD","HON","IBM","JNJ","JPM","KO","MCD","MMM","MRK",
    "MSFT","NKE","PG","TRV","UNH","V","VZ","WBA","WMT","INTC",
])

def get_universe(mode: str) -> List[str]:
    if mode == "Custom List":
        custom_str = st.session_state.get("custom_tickers", "AAPL, MSFT, TSLA, NVDA")
        return [x.strip().upper() for x in custom_str.split(",") if x.strip()]
    elif mode == "Dow Jones 30": return DOW30
    return fetch_universe()

# ── Settings init ────────────────────────────────────────────────────────────
def init_settings():
    defaults = {
        "theme":           "TradingView Dark",
        "font":            "System UI",
        "scan_list":       "S&P 500 + Nasdaq-100",
        "custom_tickers":  "AAPL, MSFT, NVDA, TSLA, SPY, QQQ",
        "auto_scan":       False,
        "alert_browser":   False,
        "alert_email":     False,
        "alert_email_addr":"",
        "smtp_user":       "",
        "smtp_pass":       "",
        "scan_interval":   15,
        "last_auto_scan":  0.0,
        "auto_top_ticker": "",
        "auto_top_score":  0.0,
        "az_ticker":       "AAPL",
        "az_period":       "1y",
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v


@st.cache_data(ttl=60 * 60)
def fetch_ohlcv(ticker: str, period: str = "1y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty: return df
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()

# ============================================================
# Technical indicators
# ============================================================
def ema(series: pd.Series, period: int) -> pd.Series: return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    ag = gain.ewm(com=period - 1, adjust=False).mean()
    al = loss.ewm(com=period - 1, adjust=False).mean()
    return 100 - (100 / (1 + ag / al.replace(0, np.nan)))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ml = ema(series, fast) - ema(series, slow)
    sl = ema(ml, signal)
    return ml, sl, ml - sl

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"]  - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()

def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up, down = df["High"].diff(), -df["Low"].diff()
    pdm = np.where((up > down) & (up > 0), up, 0.0)
    ndm = np.where((down > up) & (down > 0), down, 0.0)
    tr_s = atr(df, period)
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(com=period-1, adjust=False).mean() / tr_s.replace(0, np.nan)
    ndi = 100 * pd.Series(ndm, index=df.index).ewm(com=period-1, adjust=False).mean() / tr_s.replace(0, np.nan)
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(com=period - 1, adjust=False).mean()

def ichimoku(df: pd.DataFrame, t=9, k=26, s=52):
    ts = (df["High"].rolling(t).max() + df["Low"].rolling(t).min()) / 2
    ks = (df["High"].rolling(k).max() + df["Low"].rolling(k).min()) / 2
    sa = ((ts + ks) / 2).shift(k)
    sb = ((df["High"].rolling(s).max() + df["Low"].rolling(s).min()) / 2).shift(k)
    return ts, ks, sa, sb

def supertrend(df: pd.DataFrame, period=10, mult=3.0) -> pd.Series:
    mid = (df["High"] + df["Low"]) / 2
    atr_s = atr(df, period)
    upper, lower = mid + mult * atr_s, mid - mult * atr_s
    st_line, trend = pd.Series(np.nan, index=df.index), pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        pu, pl = upper.iloc[i-1], lower.iloc[i-1]
        upper.iloc[i] = min(upper.iloc[i], pu) if df["Close"].iloc[i-1] <= pu else upper.iloc[i]
        lower.iloc[i] = max(lower.iloc[i], pl) if df["Close"].iloc[i-1] >= pl else lower.iloc[i]
        if   df["Close"].iloc[i] > pu: trend.iloc[i] = 1
        elif df["Close"].iloc[i] < pl: trend.iloc[i] = -1
        else: trend.iloc[i] = trend.iloc[i-1]
        st_line.iloc[i] = lower.iloc[i] if trend.iloc[i] == 1 else upper.iloc[i]
    return st_line

def anchored_vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    cv = df["Volume"].cumsum()
    return (tp * df["Volume"]).cumsum() / cv.replace(0, np.nan)

def bollinger(series: pd.Series, period=20, std_mult=2.0):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    return mid + std_mult*std, mid, mid - std_mult*std

def keltner(df: pd.DataFrame, period=20, mult=1.5):
    mid = ema(df["Close"], period)
    rng = atr(df, period)
    return mid + mult*rng, mid, mid - mult*rng

def stochastic(df: pd.DataFrame, period=14, d_period=3):
    lo = df["Low"].rolling(period).min()
    hi = df["High"].rolling(period).max()
    k  = 100 * ((df["Close"] - lo) / (hi - lo).replace(0, np.nan))
    return k, k.rolling(d_period).mean()

def williams_r(df: pd.DataFrame, period=14) -> pd.Series:
    lo = df["Low"].rolling(period).min()
    hi = df["High"].rolling(period).max()
    return -100 * ((hi - df["Close"]) / (hi - lo).replace(0, np.nan))

def cci(df: pd.DataFrame, period=20) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    ma = tp.rolling(period).mean()
    md = (tp - ma).abs().rolling(period).mean()
    return (tp - ma) / (0.015 * md.replace(0, np.nan))

def aroon(df: pd.DataFrame, period=25):
    au = pd.Series(index=df.index, dtype=float)
    ad = pd.Series(index=df.index, dtype=float)
    for i in range(period - 1, len(df)):
        hh = df["High"].iloc[i-period+1:i+1].argmax()
        ll = df["Low"].iloc[i-period+1:i+1].argmin()
        au.iloc[i] = 100 * (period - (period-1-hh)) / period
        ad.iloc[i] = 100 * (period - (period-1-ll)) / period
    return au, ad

def obv(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["Close"].diff().fillna(0)) * df["Volume"]).cumsum()

def cmf(df: pd.DataFrame, period=20) -> pd.Series:
    mfm = ((df["Close"]-df["Low"]) - (df["High"]-df["Close"])) / (df["High"]-df["Low"]).replace(0, np.nan)
    mfv = mfm * df["Volume"]
    return mfv.rolling(period).sum() / df["Volume"].rolling(period).sum().replace(0, np.nan)

def mfi(df: pd.DataFrame, period=14) -> pd.Series:
    tp  = (df["High"] + df["Low"] + df["Close"]) / 3
    mf  = tp * df["Volume"]
    pos = pd.Series(np.where(tp > tp.shift(), mf, 0.0), index=df.index).rolling(period).sum()
    neg = pd.Series(np.where(tp < tp.shift(), mf, 0.0), index=df.index).rolling(period).sum()
    return 100 - (100 / (1 + pos / neg.replace(0, np.nan)))

def parabolic_sar(df: pd.DataFrame, af_start=0.02, af_step=0.02, af_max=0.2) -> pd.Series:
    hi, lo, cl = df["High"].values, df["Low"].values, df["Close"].values
    n = len(cl)
    sar = np.zeros(n); sar[0] = lo[0]
    ep = hi[0]; af = af_start; bull = True
    for i in range(1, n):
        if bull:
            sar[i] = min(sar[i-1] + af*(ep - sar[i-1]), lo[i-1], lo[i-2] if i >= 2 else lo[i-1])
            if lo[i] < sar[i]: bull=False; sar[i]=ep; ep=lo[i]; af=af_start
            elif hi[i] > ep: ep=hi[i]; af=min(af+af_step, af_max)
        else:
            sar[i] = max(sar[i-1] + af*(ep - sar[i-1]), hi[i-1], hi[i-2] if i >= 2 else hi[i-1])
            if hi[i] > sar[i]: bull=True; sar[i]=ep; ep=hi[i]; af=af_start
            elif lo[i] < ep: ep=lo[i]; af=min(af+af_step, af_max)
    return pd.Series(sar, index=df.index)

def fibonacci_levels(df: pd.DataFrame, lookback=120) -> Dict[str, float]:
    r = df.tail(lookback)
    hi, lo = r["High"].max(), r["Low"].min()
    d = hi - lo
    return {"0.236": hi-0.236*d, "0.382": hi-0.382*d, "0.5": hi-0.5*d, "0.618": hi-0.618*d, "0.786": hi-0.786*d}

def add_all_indicators(df: pd.DataFrame, rsi_period: int, macd_fast: int, macd_slow: int, macd_sig: int = 9) -> pd.DataFrame:
    x = df.copy()
    x["EMA20"]  = ema(x["Close"], 20); x["EMA50"] = ema(x["Close"], 50); x["EMA200"] = ema(x["Close"], 200)
    x["RSI"]    = rsi(x["Close"], rsi_period)
    x["ATR"]    = atr(x, 14); x["ADX"] = adx(x, 14)
    x["MACD"], x["MACD_SIG"], x["MACD_HIST"] = macd(x["Close"], macd_fast, macd_slow, macd_sig)
    x["OBV"]    = obv(x); x["CMF"] = cmf(x); x["MFI"] = mfi(x)
    x["STO_K"], x["STO_D"] = stochastic(x)
    x["WILLR"]  = williams_r(x)
    x["ROC"]    = ((x["Close"] - x["Close"].shift(12)) / x["Close"].shift(12)) * 100
    x["CCI"]    = cci(x)
    x["TRIX"]   = ema(ema(ema(x["Close"], 15), 15), 15).pct_change() * 100
    x["AROON_UP"], x["AROON_DOWN"] = aroon(x)
    x["PSAR"]   = parabolic_sar(x)
    x["SUPER"]  = supertrend(x)
    x["BB_UP"], x["BB_MID"], x["BB_LOW"] = bollinger(x["Close"])
    x["KC_UP"], x["KC_MID"], x["KC_LOW"] = keltner(x)
    x["TENKAN"], x["KIJUN"], x["SENKOU_A"], x["SENKOU_B"] = ichimoku(x)
    
    body = (x["Close"] - x["Open"]).abs()
    x["HAMMER"] = ((x[["Open","Close"]].min(axis=1) - x["Low"]) > 2 * body) & ((x["High"] - x[["Open","Close"]].max(axis=1)) < body)
    x["ENGULF"] = (x["Close"] > x["Open"]) & (x["Close"].shift() < x["Open"].shift()) & (x["Open"] <= x["Close"].shift()) & (x["Close"] >= x["Open"].shift())
    
    x["AVWAP"]  = anchored_vwap(x)
    x["DONCHIAN_HIGH"] = x["High"].rolling(20).max()
    x["DONCHIAN_LOW"]  = x["Low"].rolling(20).min()
    x["VOL_SMA20"] = x["Volume"].rolling(20).mean()
    x["GAP_UP"]    = x["Open"] > x["High"].shift(1)
    x["SQUEEZE_ON"] = (x["BB_UP"] - x["BB_LOW"]) < (x["KC_UP"] - x["KC_LOW"])
    return x

# ============================================================
# Scoring / strategy
# ============================================================
SIGNAL_WEIGHTS = {
    "vwap": 8, "obv": 4, "cmf": 4, "mfi": 3, "trend_emas": 8, "adx": 5, "ichimoku": 5, "supertrend": 5, "psar": 3,
    "rsi": 6, "macd": 6, "stoch": 3, "williams": 2, "roc": 2, "cci": 3, "trix": 2, "aroon": 3,
    "bb": 3, "keltner": 3, "atr_trap": 2, "donchian": 4, "volume_spike": 3, "gap": 1, "squeeze_release": 3, "hammer": 2, "engulf": 2, "fibonacci": 2,
}

def conviction_score(df: pd.DataFrame) -> Tuple[pd.Series, List[str]]:
    score = pd.Series(50.0, index=df.index)
    effects: Dict[str, pd.Series] = {}

    def apply(name: str, cond, pos: float, neg: Optional[float] = None):
        c = pd.Series(np.where(cond, pos, -(pos if neg is None else neg)), index=df.index)
        effects[name] = c; return c

    score += apply("vwap",       df["Close"] > df["AVWAP"],                   SIGNAL_WEIGHTS["vwap"])
    score += apply("obv",        df["OBV"] > df["OBV"].rolling(10).mean(),    SIGNAL_WEIGHTS["obv"])
    score += apply("cmf",        df["CMF"] > 0,                               SIGNAL_WEIGHTS["cmf"])
    score += apply("mfi",        df["MFI"] > 50,                              SIGNAL_WEIGHTS["mfi"])
    score += apply("trend_emas", (df["Close"] > df["EMA20"]) & (df["EMA20"] > df["EMA50"]) & (df["EMA50"] > df["EMA200"]), SIGNAL_WEIGHTS["trend_emas"])
    score += apply("adx",        df["ADX"] > 22,                              SIGNAL_WEIGHTS["adx"], SIGNAL_WEIGHTS["adx"]/2)
    score += apply("ichimoku",   df["Close"] > np.maximum(df["SENKOU_A"], df["SENKOU_B"]), SIGNAL_WEIGHTS["ichimoku"])
    score += apply("supertrend", df["Close"] > df["SUPER"],                   SIGNAL_WEIGHTS["supertrend"])
    score += apply("psar",       df["Close"] > df["PSAR"],                    SIGNAL_WEIGHTS["psar"])
    score += apply("rsi",        (df["RSI"] > 50) & (df["RSI"] < 72),         SIGNAL_WEIGHTS["rsi"])
    score += apply("macd",       df["MACD_HIST"] > 0,                         SIGNAL_WEIGHTS["macd"])
    score += apply("stoch",      df["STO_K"] > df["STO_D"],                   SIGNAL_WEIGHTS["stoch"])
    score += apply("williams",   df["WILLR"] > -50,                           SIGNAL_WEIGHTS["williams"])
    score += apply("roc",        df["ROC"] > 0,                               SIGNAL_WEIGHTS["roc"])
    score += apply("cci",        df["CCI"] > 0,                               SIGNAL_WEIGHTS["cci"])
    score += apply("trix",       df["TRIX"] > 0,                              SIGNAL_WEIGHTS["trix"])
    score += apply("aroon",      df["AROON_UP"] > df["AROON_DOWN"],           SIGNAL_WEIGHTS["aroon"])
    score += apply("bb",         df["Close"] > df["BB_MID"],                  SIGNAL_WEIGHTS["bb"])
    score += apply("keltner",    df["Close"] > df["KC_MID"],                  SIGNAL_WEIGHTS["keltner"])
    score += apply("atr_trap",   (df["ATR"]/df["Close"]).fillna(0) < (df["ATR"]/df["Close"]).fillna(0).rolling(50).mean(), SIGNAL_WEIGHTS["atr_trap"])
    score += apply("donchian",   df["Close"] > df["DONCHIAN_HIGH"].shift(1),  SIGNAL_WEIGHTS["donchian"])
    score += apply("volume_spike", df["Volume"] > 1.4*df["VOL_SMA20"],        SIGNAL_WEIGHTS["volume_spike"])
    score += apply("gap",        df["GAP_UP"],                                SIGNAL_WEIGHTS["gap"])
    score += apply("squeeze_release", (~df["SQUEEZE_ON"]) & (df["SQUEEZE_ON"].shift(1).fillna(False)) & (df["Close"] > df["EMA20"]), SIGNAL_WEIGHTS["squeeze_release"], SIGNAL_WEIGHTS["squeeze_release"]/2)
    
    effects["hammer"] = pd.Series(np.where(df["HAMMER"], SIGNAL_WEIGHTS["hammer"], 0), index=df.index); score += effects["hammer"]
    effects["engulf"] = pd.Series(np.where(df["ENGULF"], SIGNAL_WEIGHTS["engulf"], 0), index=df.index); score += effects["engulf"]
    
    fibs = fibonacci_levels(df)
    effects["fibonacci"] = pd.Series(np.where((df["Close"] - fibs["0.618"]).abs() / df["Close"] < 0.01, SIGNAL_WEIGHTS["fibonacci"], 0), index=df.index); score += effects["fibonacci"]
    
    score = score.clip(0, 100)
    ranked = sorted({k: float(v.iloc[-1]) for k, v in effects.items()}.items(), key=lambda kv: abs(kv[1]), reverse=True)
    
    rm = {"vwap":"Price above VWAP.", "trend_emas":"EMAs aligned bullish.", "macd":"MACD histogram positive.", "adx":"ADX > 22 (Trending).", "cmf":"CMF positive (inflow).", "donchian":"Price broke 20d Donchian.", "aroon":"Aroon Up dominates.", "squeeze_release":"Squeeze released upward.", "ichimoku":"Price above Cloud.", "supertrend":"SuperTrend bullish.", "rsi":"RSI in sweet spot (50-72).", "obv":"OBV rising.", "mfi":"MFI > 50 (Buying pressure).", "stoch":"Stochastic crossing up.", "psar":"PSAR confirms uptrend.", "bb":"Price above BB midline.", "keltner":"Price above Keltner midline.", "volume_spike":"Volume spike detected.", "cci":"CCI in positive momentum."}
    reasons = [rm[name] for name, val in ranked if val > 0 and name in rm][:3]
    if not reasons: reasons = ["Composite multi-factor score is balanced."]
    return score, reasons

def build_signals(df: pd.DataFrame, buy_thr=68, sell_thr=38) -> pd.Series:
    s, _ = conviction_score(df)
    sig = pd.Series(0, index=df.index)
    sig[(s > buy_thr)  & (s.shift(1) <= buy_thr)]  = 1
    sig[(s < sell_thr) & (s.shift(1) >= sell_thr)] = -1
    return sig

@dataclass
class StrategyMetrics:
    profit_factor: float
    win_rate: float
    sharpe: float
    max_drawdown: float
    total_return: float
    buy_hold_return: float

def backtest_strategy(df: pd.DataFrame, signal: pd.Series) -> StrategyMetrics:
    pos  = np.where(signal.replace(0, np.nan).ffill().shift().fillna(0) > 0, 1, 0)
    ret  = df["Close"].pct_change().fillna(0)
    sret = pd.Series(pos, index=df.index) * ret
    eq   = (1 + sret).cumprod()
    pnl  = sret[sret != 0]
    gp, gl = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return StrategyMetrics(
        profit_factor  = float((gp / gl) if gl > 0 else 999.0),
        win_rate       = float((pnl > 0).mean() if len(pnl) else 0.0),
        sharpe         = float(math.sqrt(252) * sret.mean() / sret.std() if sret.std() > 0 else 0.0),
        max_drawdown   = float((eq / eq.cummax() - 1).min()),
        total_return   = float(eq.iloc[-1] - 1),
        buy_hold_return= float((1 + ret).cumprod().iloc[-1] - 1),
    )

def optimize_params(raw: pd.DataFrame) -> Tuple[Dict[str, int], StrategyMetrics, pd.DataFrame]:
    best_obj, best_params, best_metrics, best_df = None, None, None, None
    for rp in [7, 9, 14]:
        for fast, slow in [(8,21),(12,26)]:
            tmp = add_all_indicators(raw, rp, fast, slow)
            sig = build_signals(tmp)
            m   = backtest_strategy(tmp, sig)
            obj = m.profit_factor * 0.6 + m.win_rate * 100 * 0.4
            if best_obj is None or obj > best_obj:
                best_obj, best_params, best_metrics, best_df = obj, {"rsi": rp, "macd_fast": fast, "macd_slow": slow}, m, tmp
    return best_params, best_metrics, best_df

def risk_levels(df: pd.DataFrame) -> Dict[str, float]:
    last = df.iloc[-1]
    entry, atr_v = float(last["Close"]), float(last["ATR"])
    return {
        "entry": entry, "stop": round(entry - 2 * atr_v, 2), "pt1": round(entry + 1 * atr_v, 2),
        "pt2": round(entry + 2 * atr_v, 2), "pt3": round(entry + 3 * atr_v, 2), "atr": round(atr_v, 2)
    }

# ============================================================
# Chart Builders (Theme Adapting)
# ============================================================
def build_candlestick_chart(df: pd.DataFrame, ticker: str, overlays: dict) -> go.Figure:
    th = get_theme_colors()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name=ticker, increasing_line_color="#10b981", decreasing_line_color="#ef4444",
    ))
    
    if overlays.get("ema20"): fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA20", line=dict(color="#f59e0b",width=1.5)))
    if overlays.get("ema50"): fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA50", line=dict(color="#3b82f6",width=1.5)))
    if overlays.get("ema200"): fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], name="EMA200", line=dict(color="#8b5cf6",width=1.5,dash="dash")))
    if overlays.get("vwap"): fig.add_trace(go.Scatter(x=df.index, y=df["AVWAP"], name="AVWAP", line=dict(color="#f43f5e",width=1.5,dash="dashdot")))
    
    if overlays.get("bb"):
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_UP"],  name="BB Up", line=dict(color="rgba(148,163,184,0.4)",width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_LOW"], name="BB Low", line=dict(color="rgba(148,163,184,0.4)",width=1), fill="tonexty", fillcolor="rgba(148,163,184,0.05)"))
    if overlays.get("kc"):
        fig.add_trace(go.Scatter(x=df.index, y=df["KC_UP"],  name="KC Up", line=dict(color="rgba(52,211,153,0.4)",width=1,dash="dash")))
        fig.add_trace(go.Scatter(x=df.index, y=df["KC_LOW"], name="KC Low", line=dict(color="rgba(52,211,153,0.4)",width=1,dash="dash"), fill="tonexty", fillcolor="rgba(52,211,153,0.05)"))
    if overlays.get("donchian"):
        fig.add_trace(go.Scatter(x=df.index, y=df["DONCHIAN_HIGH"], name="Donchian Hi", line=dict(color="rgba(167,139,250,0.4)",width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df["DONCHIAN_LOW"],  name="Donchian Lo", line=dict(color="rgba(167,139,250,0.4)",width=1)))
    
    if overlays.get("supertrend"): fig.add_trace(go.Scatter(x=df.index, y=df["SUPER"],  name="SuperTrend", line=dict(color="#34d399",width=1.5)))
    if overlays.get("psar"): fig.add_trace(go.Scatter(x=df.index, y=df["PSAR"], name="PSAR", mode="markers", marker=dict(symbol="circle",color="#fb923c",size=3)))
    if overlays.get("ichi"):
        fig.add_trace(go.Scatter(x=df.index, y=df["SENKOU_A"], name="Senkou A", line=dict(color="rgba(52,211,153,0.5)",width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df["SENKOU_B"], name="Senkou B", line=dict(color="rgba(248,113,113,0.5)",width=1), fill="tonexty", fillcolor="rgba(148,163,184,0.08)"))
    if overlays.get("fib"):
        colors_fib = {"0.236":"#fbbf24","0.382":"#fb923c","0.5":"#f87171","0.618":"#c084fc","0.786":"#818cf8"}
        for lv, price in fibonacci_levels(df).items():
            fig.add_hline(y=price, line_dash="dot", line_color=colors_fib.get(lv), annotation_text=f"Fib {lv}")

    fig.update_layout(
        paper_bgcolor=th["bg"], plot_bgcolor=th["chart"],
        font=dict(family=get_theme_font(), color=th["sub"]),
        title=dict(text=f"<b>{ticker}</b>", font=dict(color=th["text"],size=14)),
        xaxis=dict(showgrid=True, gridcolor=th["grid"], zeroline=False, rangeslider=dict(visible=False)),
        yaxis=dict(showgrid=True, gridcolor=th["grid"], zeroline=False),
        height=540, dragmode="pan", margin=dict(l=0,r=0,t=40,b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)")
    )
    return fig

def build_generic_oscillator(df: pd.DataFrame, traces: list, title: str, hlines: list = None, bar: bool = False, yrange=None) -> go.Figure:
    th = get_theme_colors()
    fig = go.Figure()
    for tr in traces:
        if bar: 
            fig.add_trace(go.Bar(x=df.index, y=df[tr['col']], marker_color=tr.get('colors'), name=tr['name']))
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df[tr['col']], name=tr['name'], line=dict(color=tr['color'],width=1.5)))
    
    if hlines:
        for hl in hlines: fig.add_hline(y=hl, line_dash="dash", line_color=th["sub"], line_width=1)

    fig.update_layout(
        paper_bgcolor=th["bg"], plot_bgcolor=th["chart"],
        font=dict(family=get_theme_font(), color=th["sub"]),
        title=dict(text=title, font=dict(color=th["text"],size=11)),
        xaxis=dict(showgrid=True, gridcolor=th["grid"]),
        yaxis=dict(showgrid=True, gridcolor=th["grid"], range=yrange),
        height=180, dragmode="pan", margin=dict(l=0,r=0,t=35,b=0),
        showlegend=False
    )
    return fig

def build_score_chart(df, score, signal) -> go.Figure:
    th = get_theme_colors()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=score, name="Score", fill="tozeroy", fillcolor="rgba(41,98,255,0.12)", line=dict(color="#2962ff",width=2)))
    fig.add_hline(y=72, line_dash="dash", line_color="#10b981")
    fig.add_hline(y=38, line_dash="dash", line_color="#ef4444")
    fig.add_hline(y=50, line_dash="dot",  line_color=th["sub"])
    
    buys, sells  = df.index[signal == 1], df.index[signal == -1]
    if len(buys):  fig.add_trace(go.Scatter(x=buys, y=score[buys], mode="markers", marker=dict(symbol="triangle-up", color="#10b981", size=11)))
    if len(sells): fig.add_trace(go.Scatter(x=sells, y=score[sells], mode="markers", marker=dict(symbol="triangle-down", color="#ef4444", size=11)))
    
    fig.update_layout(
        paper_bgcolor=th["bg"], plot_bgcolor=th["chart"], font=dict(family=get_theme_font(), color=th["sub"]),
        title=dict(text="Conviction Score (0-100)", font=dict(color=th["text"],size=12)),
        xaxis=dict(showgrid=True, gridcolor=th["grid"]), yaxis=dict(showgrid=True, gridcolor=th["grid"], range=[0,100]),
        height=200, dragmode="pan", margin=dict(l=0,r=0,t=35,b=0), showlegend=False
    )
    return fig


# ============================================================
# Main Scanner function
# ============================================================
@st.cache_data(ttl=30 * 60, show_spinner=False)
def scan_universe(tickers: List[str], max_scan: int = 80) -> pd.DataFrame:
    rows = []
    spy_raw = fetch_ohlcv("SPY", "6mo")
    spy_ret = float((spy_raw["Close"].iloc[-1] / spy_raw["Close"].iloc[0] - 1) * 100) if spy_raw is not None and not spy_raw.empty else 0.0

    for ticker in tickers[:max_scan]:
        try:
            raw = fetch_ohlcv(ticker, "6mo")
            if raw is None or len(raw) < 60: continue
            df = add_all_indicators(raw, 14, 12, 26)
            score_s, reasons = conviction_score(df)
            rows.append({
                "Ticker": ticker, "Price": round(df["Close"].iloc[-1], 2),
                "1D Chg%": round((df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100, 2),
                "6M Chg%": round((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100, 2),
                "RS vs SPY": round(round((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100, 2) - spy_ret, 2),
                "Score": round(score_s.iloc[-1], 1), "RSI": round(df["RSI"].iloc[-1], 1),
                "Stop Loss": round(df["Close"].iloc[-1] - 2*df["ATR"].iloc[-1], 2),
                "Verdict": "STRONG BUY" if score_s.iloc[-1]>=72 else "STRONG SELL" if score_s.iloc[-1]<=38 else "NEUTRAL",
                "Top Signal": reasons[0] if reasons else "",
            })
        except Exception: continue
    return pd.DataFrame(rows).sort_values(["Score","RS vs SPY"], ascending=False).reset_index(drop=True)

# ── Alerts & Auto-Scan Helpers ──────────────────────────────────────────────
def send_email_alert(to_addr: str, smtp_user: str, smtp_pass: str, subject: str, body: str) -> bool:
    try:
        msg = MIMEText(body, "plain")
        msg["Subject"] = subject
        msg["From"]    = smtp_user
        msg["To"]      = to_addr
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as s:
            s.starttls()
            s.login(smtp_user, smtp_pass)
            s.sendmail(smtp_user, to_addr, msg.as_string())
        return True
    except Exception:
        return False

def push_browser_notification(title: str, body: str):
    components.html(f"""<script>
    (function(){{
      if(Notification.permission==='granted'){{
        new Notification({repr(title)},{{body:{repr(body)},icon:'https://cdn-icons-png.flaticon.com/32/2168/2168252.png'}});
      }}
    }})();
    </script>""", height=0)

def check_auto_scan(universe: List[str]):
    if not st.session_state.get("auto_scan", False):
        return
    interval = st.session_state.get("scan_interval", 15) * 60
    now = time.time()
    last = st.session_state.get("last_auto_scan", 0.0)
    if now - last < interval:
        return
    st.session_state["last_auto_scan"] = now
    with st.spinner("Auto-scan running…"):
        results = scan_universe(tuple(universe[:80]), 80)
    if results.empty:
        return
    buys = results[results["Verdict"] == "STRONG BUY"]
    top = buys.iloc[0] if len(buys) > 0 else results.iloc[0]
    st.session_state["auto_top_ticker"] = top["Ticker"]
    st.session_state["auto_top_score"]  = top["Score"]
    title = f"Trading Alert — {top['Ticker']}"
    body  = f"{top['Verdict']} · Score {top['Score']}/100 · ${top['Price']:.2f}"
    if st.session_state.get("alert_browser"):
        push_browser_notification(title, body)
    if st.session_state.get("alert_email") and st.session_state.get("alert_email_addr"):
        send_email_alert(
            st.session_state["alert_email_addr"],
            st.session_state.get("smtp_user", ""),
            st.session_state.get("smtp_pass", ""),
            title, body,
        )

# ============================================================
# Application Entry
# ============================================================
_PCFG = {"scrollZoom": True, "displayModeBar": False}

def main():
    init_settings()
    inject_theme()
    universe = get_universe(st.session_state["scan_list"])
    check_auto_scan(universe)

    top_t = st.session_state.get("auto_top_ticker", "")
    if top_t:
        st.info(f"Auto-scan alert: **{top_t}** is the top pick right now (Score {st.session_state.get('auto_top_score', 0)}/100).")

    tab_analyze, tab_scan, tab_backtest, tab_settings = st.tabs(["📈 ANALYZE", "🔍 SCANNER", "📊 BACKTEST", "⚙️ SETTINGS"])

    # ════════════════════════════════════════════════════════
    # Tab 1 — Analyze
    # ════════════════════════════════════════════════════════
    with tab_analyze:
        left, right = st.columns([1, 3], gap="large")
        with left:
            ticker_input = st.text_input("Ticker Symbol", value=st.session_state.get("az_ticker", "AAPL"), help="Type any valid stock ticker.")
            period = st.selectbox("Historical Data", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
            
            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
            with st.expander("🛠️ Chart Overlays & Oscillators", expanded=False):
                st.markdown("**Price Overlays**")
                ov_ema20 = st.checkbox("EMA 20", True)
                ov_ema50 = st.checkbox("EMA 50", True)
                ov_ema200 = st.checkbox("EMA 200", False)
                ov_vwap = st.checkbox("Anchored VWAP", True)
                ov_bb = st.checkbox("Bollinger Bands", False)
                ov_kc = st.checkbox("Keltner Channels", False)
                ov_donchian = st.checkbox("Donchian Channels", False)
                ov_super = st.checkbox("SuperTrend", True)
                ov_ichi = st.checkbox("Ichimoku Cloud", False)
                ov_psar = st.checkbox("Parabolic SAR", False)
                ov_fib = st.checkbox("Fibonacci Levels", False)
                
                st.markdown("**Oscillators (Sub-charts)**")
                osc_vol = st.checkbox("Volume", True)
                osc_rsi = st.checkbox("RSI", True)
                osc_macd = st.checkbox("MACD", True)
                osc_stoch = st.checkbox("Stochastic", False)
                osc_willr = st.checkbox("Williams %R", False)
                osc_obv = st.checkbox("On-Balance Volume", False)
                osc_cmf = st.checkbox("Chaikin Money Flow", False)
                osc_mfi = st.checkbox("Money Flow Index", False)
                osc_adx = st.checkbox("ADX", False)
                osc_cci = st.checkbox("CCI", False)
                osc_aroon = st.checkbox("Aroon", False)

            overlays_dict = {
                "ema20": ov_ema20, "ema50": ov_ema50, "ema200": ov_ema200, "vwap": ov_vwap, 
                "bb": ov_bb, "kc": ov_kc, "donchian": ov_donchian, "supertrend": ov_super, 
                "ichi": ov_ichi, "psar": ov_psar, "fib": ov_fib
            }

            st.markdown('<div style="margin-top:14px"></div>', unsafe_allow_html=True)
            run_btn = st.button("RUN ANALYSIS", type="primary", use_container_width=True)

        with right:
            if run_btn:
                st.session_state["az_ticker"] = ticker_input.strip().upper()
                st.session_state["az_period"] = period
            
            t = st.session_state.get("az_ticker", "AAPL")
            p = st.session_state.get("az_period", "1y")

            with st.spinner(f"Analyzing {t}…"):
                raw = fetch_ohlcv(t, p)
                if raw is None or raw.empty:
                    st.error(f"Could not load data for **{t}**. Ensure the ticker is valid.")
                else:
                    raw2y = fetch_ohlcv(t, "2y")
                    bp = optimize_params(raw2y)[0] if (raw2y is not None and len(raw2y)>120) else {"rsi": 14, "macd_fast": 12, "macd_slow": 26}
                    df = add_all_indicators(raw, bp["rsi"], bp["macd_fast"], bp["macd_slow"])
                    score_series, reasons = conviction_score(df)
                    signal = build_signals(df)
                    rl = risk_levels(df)
                    last, last_score = df.iloc[-1], float(score_series.iloc[-1])
                    verdict = "STRONG BUY" if last_score>=72 else "STRONG SELL" if last_score<=38 else "NEUTRAL"

                    cls_map = {"STRONG BUY": "verdict-buy", "STRONG SELL": "verdict-sell", "NEUTRAL": "verdict-neutral"}
                    st.markdown(f'<div class="verdict-card {cls_map.get(verdict,"verdict-neutral")}">{t} &nbsp;·&nbsp; {verdict} &nbsp;·&nbsp; {last_score:.1f} / 100</div>', unsafe_allow_html=True)

                    m1, m2, m3, m4, m5, m6 = st.columns(6)
                    m1.metric("Price", f"${last['Close']:.2f}", f"{(last['Close'] / df['Close'].iloc[-2] - 1) * 100:+.2f}%")
                    m2.metric("RSI", f"{last['RSI']:.1f}")
                    m3.metric("ATR", f"{last['ATR']:.2f}")
                    m4.metric("ADX", f"{last['ADX']:.1f}")
                    m5.metric("CMF", f"{last['CMF']:.3f}")
                    m6.metric("MFI", f"{last['MFI']:.1f}")

                    st.markdown('<div style="font-family:monospace;font-size:0.75rem;color:#787b86;margin:16px 0 6px 0">SYSTEM CONVICTION DRIVERS</div>', unsafe_allow_html=True)
                    for i, r in enumerate(reasons, 1):
                        st.markdown(f'<div class="reason-box">{i}. {r}</div>', unsafe_allow_html=True)

                    pct_stop = abs(rl["entry"] - rl["stop"]) / rl["entry"] * 100
                    st.markdown(f"""
                    <div class="risk-card" style="margin-top:14px">
                      <div class="risk-row"><span class="risk-label">Entry (last close)</span><span style="color:#e6edf3;font-weight:700;">${rl['entry']:.2f}</span></div>
                      <div class="risk-row"><span class="risk-label">🛑 Stop Loss (2 ATR)</span><span class="risk-stop">${rl['stop']:.2f} <span style="font-size:0.8rem">(-{pct_stop:.1f}%)</span></span></div>
                      <div class="risk-row"><span class="risk-label">🎯 Target 1 (1 ATR)</span><span class="risk-pt1">${rl['pt1']:.2f}</span></div>
                      <div class="risk-row"><span class="risk-label">🎯 Target 2 (2 ATR)</span><span class="risk-pt2">${rl['pt2']:.2f}</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Charts
                    st.markdown('<div style="height:15px"></div>', unsafe_allow_html=True)
                    fig_price = build_candlestick_chart(df, t, overlays_dict)
                    buys_px, sells_px = df.index[signal == 1], df.index[signal == -1]
                    if len(buys_px):  fig_price.add_trace(go.Scatter(x=buys_px, y=df["Low"][buys_px]*0.99, mode="markers", name="BUY", marker=dict(symbol="triangle-up", color="#10b981", size=11)))
                    if len(sells_px): fig_price.add_trace(go.Scatter(x=sells_px, y=df["High"][sells_px]*1.01, mode="markers", name="SELL", marker=dict(symbol="triangle-down", color="#ef4444", size=11)))
                    st.plotly_chart(fig_price, use_container_width=True, config=_PCFG)
                    
                    st.plotly_chart(build_score_chart(df, score_series, signal), use_container_width=True, config=_PCFG)

                    # Render toggled oscillators
                    if osc_vol:
                        colors = ["#10b981" if c >= o else "#ef4444" for c, o in zip(df["Close"], df["Open"])]
                        st.plotly_chart(build_generic_oscillator(df, [{"col":"Volume","name":"Vol","colors":colors}, {"col":"VOL_SMA20","name":"SMA20","color":"#f59e0b"}], "Volume & SMA20", bar=True), use_container_width=True, config=_PCFG)
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        if osc_rsi: st.plotly_chart(build_generic_oscillator(df, [{"col":"RSI", "name":"RSI", "color":"#fbbf24"}], "RSI (14)", hlines=[30, 50, 70], yrange=[0,100]), use_container_width=True, config=_PCFG)
                        if osc_stoch: st.plotly_chart(build_generic_oscillator(df, [{"col":"STO_K", "name":"%K", "color":"#818cf8"}, {"col":"STO_D", "name":"%D", "color":"#fb923c"}], "Stochastic Momentum", hlines=[20, 80], yrange=[0,100]), use_container_width=True, config=_PCFG)
                        if osc_obv: st.plotly_chart(build_generic_oscillator(df, [{"col":"OBV", "name":"OBV", "color":"#34d399"}], "On-Balance Volume (OBV)"), use_container_width=True, config=_PCFG)
                        if osc_mfi: st.plotly_chart(build_generic_oscillator(df, [{"col":"MFI", "name":"MFI", "color":"#f472b6"}], "Money Flow Index (MFI)", hlines=[20, 80], yrange=[0,100]), use_container_width=True, config=_PCFG)
                        if osc_cci: st.plotly_chart(build_generic_oscillator(df, [{"col":"CCI", "name":"CCI", "color":"#a78bfa"}], "Commodity Channel Index (CCI)", hlines=[-100, 0, 100]), use_container_width=True, config=_PCFG)
                    with c2:
                        if osc_macd:
                            fig_macd = build_generic_oscillator(df, [{"col":"MACD", "name":"MACD", "color":"#3b82f6"}, {"col":"MACD_SIG", "name":"Signal", "color":"#f43f5e"}], "MACD")
                            colors_macd = ["#10b981" if v >= 0 else "#ef4444" for v in df["MACD_HIST"]]
                            fig_macd.add_trace(go.Bar(x=df.index, y=df["MACD_HIST"], marker_color=colors_macd, name="Hist"))
                            st.plotly_chart(fig_macd, use_container_width=True, config=_PCFG)
                        if osc_willr: st.plotly_chart(build_generic_oscillator(df, [{"col":"WILLR", "name":"%R", "color":"#34d399"}], "Williams %R", hlines=[-20, -80], yrange=[-100,0]), use_container_width=True, config=_PCFG)
                        if osc_cmf: st.plotly_chart(build_generic_oscillator(df, [{"col":"CMF", "name":"CMF", "color":"#10b981"}], "Chaikin Money Flow (CMF)", hlines=[0]), use_container_width=True, config=_PCFG)
                        if osc_adx: st.plotly_chart(build_generic_oscillator(df, [{"col":"ADX", "name":"ADX", "color":"#f59e0b"}], "Average Directional Index (ADX)", hlines=[20, 25]), use_container_width=True, config=_PCFG)
                        if osc_aroon: st.plotly_chart(build_generic_oscillator(df, [{"col":"AROON_UP", "name":"Up", "color":"#10b981"}, {"col":"AROON_DOWN", "name":"Down", "color":"#ef4444"}], "Aroon Oscillator", hlines=[50]), use_container_width=True, config=_PCFG)

    # ════════════════════════════════════════════════════════
    # Tab 2 — Scanner
    # ════════════════════════════════════════════════════════
    with tab_scan:
        sc1, sc2 = st.columns([3, 1], gap="medium")
        with sc1: max_scan = st.slider("Stocks to scan", 10, 150, 50, 10)
        with sc2:
            st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
            scan_btn = st.button("RUN SCANNER", type="primary", use_container_width=True)

        if scan_btn:
            with st.spinner(f"Scanning {min(int(max_scan), len(universe))} stocks from {st.session_state['scan_list']}…"):
                results = scan_universe(tuple(universe), int(max_scan))
            if not results.empty:
                buys, sells = results[results["Verdict"] == "STRONG BUY"], results[results["Verdict"] == "STRONG SELL"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Scanned", len(results))
                c2.metric("Strong Buys", len(buys))
                c3.metric("Strong Sells", len(sells))
                c4.metric("Neutral", len(results) - len(buys) - len(sells))

                st.markdown('<div style="font-family:monospace;font-size:0.75rem;color:#10b981;margin:22px 0 10px 0">TOP SETUPS</div>', unsafe_allow_html=True)
                top5 = buys.head(5) if len(buys) >= 1 else results.head(5)
                pick_cols = st.columns(min(len(top5), 5))
                for col, (_, row) in zip(pick_cols, top5.iterrows()):
                    col.markdown(f"""
                    <div class="pick-card">
                      <div class="pick-ticker">{row['Ticker']}</div>
                      <div class="pick-score">{row['Score']} <span style="color:#787b86;font-size:0.65rem">/ 100</span></div>
                      <div style="color:#d1d4dc;font-size:0.9rem;margin:6px 0">${row['Price']:.2f}</div>
                      <div style="color:{'#10b981' if row['1D Chg%'] >= 0 else '#ef4444'};font-size:0.85rem;font-weight:600;">
                        {'▲' if row['1D Chg%'] >= 0 else '▼'} {abs(row['1D Chg%']):.2f}% today
                      </div>
                    </div>""", unsafe_allow_html=True)

                st.markdown('<div style="font-family:monospace;font-size:0.75rem;color:#787b86;margin:26px 0 8px 0">ALL RESULTS</div>', unsafe_allow_html=True)
                def _style_row(row):
                    styles = [""] * len(row)
                    idx = list(row.index)
                    if "Verdict" in idx: styles[idx.index("Verdict")] = "color:#10b981;font-weight:700" if row["Verdict"] == "STRONG BUY" else "color:#ef4444;font-weight:700" if row["Verdict"] == "STRONG SELL" else "color:#f59e0b"
                    if "Score" in idx and row["Score"] >= 72: styles[idx.index("Score")] = "color:#10b981;font-weight:700"
                    if "RS vs SPY" in idx: styles[idx.index("RS vs SPY")] = "color:#10b981" if row["RS vs SPY"] > 0 else "color:#ef4444"
                    return styles
                st.dataframe(results.drop(columns=["Top Signal"], errors="ignore").style.apply(_style_row, axis=1), use_container_width=True, height=480)

    # ════════════════════════════════════════════════════════
    # Tab 3 — Backtest
    # ════════════════════════════════════════════════════════
    with tab_backtest:
        bx, by = st.columns([1, 2], gap="large")
        with bx:
            bt_ticker = st.text_input("Ticker to Test", value="AAPL", key="bt_t", help="Enter symbol to run strategy against.")
            bt_ticker = bt_ticker.strip().upper()
            start_cap = st.number_input("Starting Capital ($)", min_value=100.0, value=500.0, step=100.0)
            bt_period = st.selectbox("Historical Duration", ["1y", "2y", "5y"], index=1, key="bt_p")
            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
            run_bt = st.button("RUN DOLLAR SIMULATION", type="primary", use_container_width=True)

        with by:
            if run_bt:
                with st.spinner(f"Fetching data & Running Parameter Realignment for {bt_ticker}..."):
                    raw = fetch_ohlcv(bt_ticker, bt_period)
                if raw is None or raw.empty:
                    st.error(f"Could not load data for {bt_ticker}.")
                else:
                    bp, metrics, df_bt = optimize_params(raw)
                    sig_bt = build_signals(df_bt)
                    
                    st.success(f"⚙️ **Realignment Complete:** Calibrated strictly for **{bt_ticker}**. Optimal Settings Applied: RSI={bp['rsi']}, MACD=({bp['macd_fast']}, {bp['macd_slow']})")

                    pos = np.where(sig_bt.replace(0, np.nan).ffill().shift().fillna(0) > 0, 1, 0)
                    ret = df_bt["Close"].pct_change().fillna(0)
                    s_eq_curve = start_cap * (1 + pd.Series(pos, index=df_bt.index) * ret).cumprod()
                    bh_eq_curve = start_cap * (1 + ret).cumprod()

                    final_bal = float(s_eq_curve.iloc[-1])
                    net_profit = final_bal - start_cap
                    bh_final = float(bh_eq_curve.iloc[-1])

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Final Balance", f"${final_bal:,.2f}", f"${net_profit:+,.2f} Net Profit", help=f"Started with ${start_cap:,.2f}")
                    m2.metric("Win Rate", f"{metrics.win_rate:.1%}")
                    m3.metric("Profit Factor", f"{metrics.profit_factor:.2f}")

                    m4, m5, m6 = st.columns(3)
                    m4.metric("Strategy Total Return", f"{metrics.total_return:.1%}")
                    m5.metric("Buy & Hold Return", f"{metrics.buy_hold_return:.1%}", help=f"B&H Final Balance: ${bh_final:,.2f}")
                    m6.metric("Max Drawdown", f"{metrics.max_drawdown:.1%}")

                    # Dollar Equity Curve Chart
                    th = get_theme_colors()
                    eq_fig = go.Figure()
                    eq_fig.add_trace(go.Scatter(x=df_bt.index, y=s_eq_curve,  name="Strategy ($)", line=dict(color="#2962ff", width=2.5)))
                    eq_fig.add_trace(go.Scatter(x=df_bt.index, y=bh_eq_curve, name="Buy & Hold ($)", line=dict(color=th["sub"], width=1.5, dash="dash")))
                    
                    buys_i, sells_i = df_bt.index[sig_bt == 1], df_bt.index[sig_bt == -1]
                    if len(buys_i):  eq_fig.add_trace(go.Scatter(x=buys_i,  y=s_eq_curve[buys_i],  mode="markers", name="Buy",  marker=dict(symbol="triangle-up",  color="#10b981", size=10)))
                    if len(sells_i): eq_fig.add_trace(go.Scatter(x=sells_i, y=s_eq_curve[sells_i], mode="markers", name="Sell", marker=dict(symbol="triangle-down", color="#ef4444", size=10)))
                    
                    eq_fig.update_layout(
                        paper_bgcolor=th["bg"], plot_bgcolor=th["chart"], font=dict(family=get_theme_font(), color=th["sub"]),
                        title=dict(text=f"<b>{bt_ticker}</b> — Simulated $ Equity Curve", font=dict(color=th["text"], size=13)),
                        xaxis=dict(showgrid=True, gridcolor=th["grid"]), yaxis=dict(showgrid=True, gridcolor=th["grid"]),
                        height=400, dragmode="pan", margin=dict(l=0, r=0, t=50, b=0),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
                    )
                    st.plotly_chart(eq_fig, use_container_width=True, config=_PCFG)

    # ════════════════════════════════════════════════════════
    # Tab 4 — Settings
    # ════════════════════════════════════════════════════════
    with tab_settings:
        s1, s2 = st.columns(2, gap="large")
        with s1:
            st.markdown("#### Interface & Appearance")
            theme_choice = st.selectbox("Color Theme", list(_THEMES.keys()), index=list(_THEMES.keys()).index(st.session_state.get("theme", "TradingView Dark")))
            if theme_choice != st.session_state["theme"]: st.session_state["theme"] = theme_choice; st.rerun()

            font_choice = st.selectbox("Typography", ["System UI", "Monospace", "Sans-serif", "Serif"], index=["System UI", "Monospace", "Sans-serif", "Serif"].index(st.session_state.get("font", "System UI")))
            if font_choice != st.session_state["font"]: st.session_state["font"] = font_choice; st.rerun()

            st.markdown('<div style="margin-top:20px"></div>', unsafe_allow_html=True)
            st.markdown("#### Scanner Universe Setup")
            scan_list_choice = st.selectbox("Market Index", ["S&P 500 + Nasdaq-100", "S&P 500", "Nasdaq-100", "Dow Jones 30", "Custom List"], index=["S&P 500 + Nasdaq-100", "S&P 500", "Nasdaq-100", "Dow Jones 30", "Custom List"].index(st.session_state.get("scan_list", "S&P 500 + Nasdaq-100")))
            if scan_list_choice != st.session_state["scan_list"]: st.session_state["scan_list"] = scan_list_choice; st.rerun()

            if scan_list_choice == "Custom List":
                custom_ticks = st.text_area("Custom Tickers (comma separated)", value=st.session_state.get("custom_tickers", "AAPL, MSFT, TSLA, NVDA"), help="Enter your personal watchlist.")
                st.session_state["custom_tickers"] = custom_ticks

        with s2:
            st.markdown("#### Alerts & Auto-Scan")
            auto_on = st.toggle("Enable auto-scan", value=st.session_state.get("auto_scan", False))
            st.session_state["auto_scan"] = auto_on

            if auto_on:
                interval = st.select_slider("Scan interval", options=[5, 10, 15, 30, 60], value=st.session_state.get("scan_interval", 15), format_func=lambda x: f"{x} min")
                st.session_state["scan_interval"] = interval
                last_ts = st.session_state.get("last_auto_scan", 0.0)
                st.caption(f"Last scan: {int((time.time() - last_ts) / 60)} min ago" if last_ts > 0 else "No auto-scan has run yet.")

                st.markdown('<div style="margin-top:12px"></div>', unsafe_allow_html=True)
                br_on = st.checkbox("Browser notifications", value=st.session_state.get("alert_browser", False))
                st.session_state["alert_browser"] = br_on
                if br_on:
                    components.html("""<div style="margin-top:6px"><button onclick="Notification.requestPermission().then(p=>{document.getElementById('ns').textContent = p==='granted' ? '✓ Notifications enabled' : '✗ Blocked'; document.getElementById('ns').style.color = p==='granted' ? '#34d399' : '#f87171';});" style="background:#2962ff;color:#fff;border:none;border-radius:6px;padding:8px 18px;font-family:monospace;cursor:pointer;">Request Permission</button><span id="ns" style="margin-left:12px;font-family:monospace;font-size:0.8rem;color:#787b86;"></span></div>""", height=46)

                email_on = st.checkbox("Email alerts", value=st.session_state.get("alert_email", False))
                st.session_state["alert_email"] = email_on
                if email_on:
                    ea1, ea2 = st.columns(2)
                    with ea1: st.session_state["alert_email_addr"] = st.text_input("Send to", value=st.session_state.get("alert_email_addr", ""))
                    with ea2: st.session_state["smtp_user"] = st.text_input("Gmail username", value=st.session_state.get("smtp_user", ""))
                    st.session_state["smtp_pass"] = st.text_input("Gmail App Password", value=st.session_state.get("smtp_pass", ""), type="password")
                    if st.button("Send test email"):
                        if send_email_alert(st.session_state["alert_email_addr"], st.session_state["smtp_user"], st.session_state["smtp_pass"], "Trading Terminal", "Test working."): st.success("Sent.")
                        else: st.error("Failed to send.")

if __name__ == "__main__":
    main()
