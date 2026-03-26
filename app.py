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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

st.set_page_config(
    page_title="Trading Terminal Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1rem !important; padding-bottom: 1rem !important; max-width: 100% !important; }
  .stApp { background-color: #131722; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] { background-color: #1e222d; border: 1px solid #2a2e39; border-radius: 8px; padding: 4px; gap: 4px; }
  .stTabs [data-baseweb="tab"] { color: #787b86; font-size: 0.85rem; letter-spacing: 0.05em; padding: 8px 20px; border-radius: 6px; border: none !important; }
  .stTabs [aria-selected="true"] { background-color: #131722 !important; color: #2962ff !important; border-bottom: 2px solid #2962ff !important; font-weight: 700; }

  /* ── Metric cards ── */
  [data-testid="stMetric"] { background-color: #1e222d; border: 1px solid #2a2e39; border-radius: 8px; padding: 14px 18px; box-shadow: 0 4px 6px rgba(0,0,0,0.2); }
  [data-testid="stMetricLabel"] { color: #787b86 !important; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; }
  [data-testid="stMetricValue"] { color: #d1d4dc !important; font-size: 1.3rem; font-weight: 700; }
  
  /* ── Verdict banner ── */
  .verdict-card { padding: 14px 24px; border-radius: 8px; font-size: 1.35rem; font-weight: 800; text-align: center; margin-bottom: 14px; letter-spacing: 0.04em; }
  .verdict-buy     { background: linear-gradient(135deg,#0d2e1e,#0d3d28); border: 1px solid #089981; color: #26a69a; }
  .verdict-sell    { background: linear-gradient(135deg,#2e0d0d,#3d1010); border: 1px solid #f23645; color: #ef5350; }
  .verdict-neutral { background: linear-gradient(135deg,#1a1d27,#1e222d); border: 1px solid #2a2e39; color: #787b86; }

  /* ── Risk & News Cards ── */
  .risk-card, .news-card { background-color: #1e222d; border: 1px solid #2a2e39; border-radius: 8px; padding: 14px 18px; margin-top: 10px; }
  .news-card { transition: border-color 0.2s; cursor: pointer; }
  .news-card:hover { border-color: #2962ff; }
  .news-title { font-size: 1rem; font-weight: 700; color: #d1d4dc; margin-bottom: 6px; }
  .news-meta { font-size: 0.75rem; color: #787b86; }
  
  .risk-row { display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid #2a2e39; }
  .risk-row:last-child { border-bottom: none; }
  .risk-label { color: #787b86; font-size: 0.8rem; }
  
  /* ── Utilities ── */
  .stButton > button { background-color: #2962ff !important; color: #fff !important; border: none !important; border-radius: 6px !important; font-weight: 700 !important; }
  .stButton > button:hover { background-color: #1e53e5 !important; }
  hr { border-color: #2a2e39; margin: 14px 0; }
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #1e222d; }
  ::-webkit-scrollbar-thumb { background: #2a2e39; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Core Engine & Initialization
# ============================================================
def init_settings():
    defaults = {
        "theme": "TradingView", "font": "JetBrains Mono",
        "scan_list": "Major ETFs & Funds", "custom_tickers": "",
        "active_tabs": ["SPY", "QQQ"], "az_period": "6mo",
        "starting_capital": 5000.0, "risk_per_trade": 1.0,
        "portfolio": {}, "trade_history": [], "cash": 5000.0,
        "ov_ema20": True, "ov_ema50": True, "ov_ema200": True, "ov_avwap": True, "ov_bb": True, "ov_super": True, "ov_ichi": False, "ov_fib": False, "ov_psar": False,
        "sc_rsi": True, "sc_macd": True, "sc_vol": True, "sc_stoch": False, "sc_willr": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Auto-sync query params to session state for basic persistence
    q_params = st.query_params
    if "tabs" in q_params:
        st.session_state["active_tabs"] = q_params["tabs"].split(",")

def save_tabs_state():
    st.query_params["tabs"] = ",".join(st.session_state["active_tabs"])

# Universe Definitions
ETFS = sorted(["SPY", "QQQ", "DIA", "IWM", "VTI", "VOO", "ARKK", "GLD", "SLV", "USO", "UNG", "TLT", "TMF", "XLF", "XLK", "XLE", "XLU", "XLV", "XLY", "XLP", "XLI", "XLB", "XLRE"])
DOW30 = sorted(["AAPL","AMGN","AXP","BA","CAT","CRM","CSCO","CVX","DIS","DOW","GS","HD","HON","IBM","JNJ","JPM","KO","MCD","MMM","MRK","MSFT","NKE","PG","TRV","UNH","V","VZ","WBA","WMT","INTC"])

@st.cache_data(ttl=3600*24)
def fetch_sp500_only() -> List[str]:
    try:
        r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        return sorted([s.replace(".", "-").strip().upper() for s in pd.read_html(StringIO(r.text))[0]["Symbol"].astype(str).tolist()])
    except: return DOW30

@st.cache_data(ttl=3600*24)
def fetch_ndx100_only() -> List[str]:
    try:
        r = requests.get("https://en.wikipedia.org/wiki/Nasdaq-100", headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        for tbl in pd.read_html(StringIO(r.text)):
            cols = [str(c).strip().lower() for c in tbl.columns]
            if "ticker" in cols: return sorted([s.strip().upper() for s in tbl[tbl.columns[cols.index("ticker")]].astype(str).tolist()])
    except: return DOW30
    return DOW30

def get_universe(mode: str) -> List[str]:
    if mode == "Custom List":
        parsed = [t.strip().upper() for t in st.session_state.get("custom_tickers", "").replace(","," ").split() if t.strip()]
        return parsed if parsed else ETFS
    if mode == "Major ETFs & Funds": return ETFS
    if mode == "S&P 500": return fetch_sp500_only()
    if mode == "Nasdaq-100": return fetch_ndx100_only()
    if mode == "Dow Jones 30": return DOW30
    return list(set(fetch_sp500_only() + fetch_ndx100_only()))

# Data Fetching (Always fetch 5y for infinite scroll)
@st.cache_data(ttl=60 * 30)
def fetch_ohlcv(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="5y", auto_adjust=True, progress=False)
    if df.empty: return df
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()

@st.cache_data(ttl=60 * 15)
def fetch_news(ticker: str) -> List[dict]:
    try:
        t = yf.Ticker(ticker)
        return t.news[:8]
    except: return []

# ============================================================
# Technical Indicators (Optimized)
# ============================================================
def ema(s: pd.Series, p: int) -> pd.Series: return s.ewm(span=p, adjust=False).mean()
def atr(df: pd.DataFrame, p: int = 14) -> pd.Series:
    tr = pd.concat([df["High"]-df["Low"], (df["High"]-df["Close"].shift()).abs(), (df["Low"]-df["Close"].shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(com=p-1, adjust=False).mean()

def add_all_indicators(df: pd.DataFrame, rsi_p: int=14, macd_f: int=12, macd_s: int=26) -> pd.DataFrame:
    x = df.copy()
    x["EMA20"] = ema(x["Close"], 20); x["EMA50"] = ema(x["Close"], 50); x["EMA200"] = ema(x["Close"], 200)
    
    # RSI
    delta = x["Close"].diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    ag = gain.ewm(com=rsi_p-1, adjust=False).mean(); al = loss.ewm(com=rsi_p-1, adjust=False).mean()
    x["RSI"] = 100 - (100 / (1 + ag / al.replace(0, np.nan)))
    
    # MACD
    ml = ema(x["Close"], macd_f) - ema(x["Close"], macd_s); sl = ema(ml, 9)
    x["MACD"] = ml; x["MACD_SIG"] = sl; x["MACD_HIST"] = ml - sl
    
    x["ATR"] = atr(x, 14)
    x["VOL_SMA20"] = x["Volume"].rolling(20).mean()
    
    # Anchored VWAP (approx 1y)
    tp = (x["High"]+x["Low"]+x["Close"])/3
    x["AVWAP"] = (tp * x["Volume"]).rolling(252, min_periods=1).sum() / x["Volume"].rolling(252, min_periods=1).sum().replace(0, np.nan)
    
    # SuperTrend
    mid = (x["High"]+x["Low"])/2; atr_s = x["ATR"] * 3.0
    upper = mid + atr_s; lower = mid - atr_s
    st_line = pd.Series(np.nan, index=x.index); trend = pd.Series(1, index=x.index)
    for i in range(1, len(x)):
        pu, pl = upper.iloc[i-1], lower.iloc[i-1]
        upper.iloc[i] = min(upper.iloc[i], pu) if x["Close"].iloc[i-1] <= pu else upper.iloc[i]
        lower.iloc[i] = max(lower.iloc[i], pl) if x["Close"].iloc[i-1] >= pl else lower.iloc[i]
        if x["Close"].iloc[i] > pu: trend.iloc[i] = 1
        elif x["Close"].iloc[i] < pl: trend.iloc[i] = -1
        else: trend.iloc[i] = trend.iloc[i-1]
        st_line.iloc[i] = lower.iloc[i] if trend.iloc[i] == 1 else upper.iloc[i]
    x["SUPER"] = st_line
    
    # Stoch & Williams
    lo = x["Low"].rolling(14).min(); hi = x["High"].rolling(14).max()
    x["STO_K"] = 100 * ((x["Close"] - lo) / (hi - lo).replace(0, np.nan))
    x["STO_D"] = x["STO_K"].rolling(3).mean()
    x["WILLR"] = -100 * ((hi - x["Close"]) / (hi - lo).replace(0, np.nan))
    
    # Bollinger
    mid_bb = x["Close"].rolling(20).mean(); std_bb = x["Close"].rolling(20).std()
    x["BB_UP"] = mid_bb + 2*std_bb; x["BB_LOW"] = mid_bb - 2*std_bb; x["BB_MID"] = mid_bb
    
    # Ichimoku Basic
    x["TENKAN"] = (x["High"].rolling(9).max() + x["Low"].rolling(9).min()) / 2
    x["KIJUN"] = (x["High"].rolling(26).max() + x["Low"].rolling(26).min()) / 2
    x["SENKOU_A"] = ((x["TENKAN"] + x["KIJUN"]) / 2).shift(26)
    x["SENKOU_B"] = ((x["High"].rolling(52).max() + x["Low"].rolling(52).min()) / 2).shift(26)
    
    return x

# ============================================================
# Strategy & Optimization
# ============================================================
def conviction_score(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    score = pd.Series(50.0, index=df.index)
    
    # Vectorized scoring logic
    score += np.where(df["Close"] > df["AVWAP"], 8, -8)
    score += np.where((df["Close"] > df["EMA20"]) & (df["EMA20"] > df["EMA50"]), 8, -8)
    score += np.where((df["RSI"] > 50) & (df["RSI"] < 70), 6, -6)
    score += np.where(df["MACD_HIST"] > 0, 6, -6)
    score += np.where(df["Close"] > df["SUPER"], 5, -5)
    score += np.where(df["Close"] > df["BB_MID"], 3, -3)
    score += np.where(df["STO_K"] > df["STO_D"], 3, -3)
    score += np.where(df["Volume"] > 1.2*df["VOL_SMA20"], 3, 0)
    
    score = score.clip(0, 100)
    
    last = df.iloc[-1]
    reason = "Bullish momentum aligned." if score.iloc[-1] > 60 else "Bearish pressure dominant." if score.iloc[-1] < 40 else "Neutral range-bound action."
    if last["MACD_HIST"] > 0 and last["Close"] > last["EMA50"]: reason = "MACD accelerating above primary trend EMA50."
    elif last["RSI"] > 70: reason = "Warning: RSI indicates overbought conditions."
    
    return score, reason

def fast_optimize(raw: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
    """Tests a minimal grid to find the best parameters for THIS specific stock quickly."""
    best_pf, best_p, best_df = 0, {"rsi":14, "macd_f":12, "macd_s":26}, raw
    for rp in [10, 14]:
        for mf, ms in [(8,21), (12,26)]:
            tmp = add_all_indicators(raw, rp, mf, ms)
            sc, _ = conviction_score(tmp)
            sig = np.where((sc > 68) & (sc.shift(1) <= 68), 1, np.where((sc < 38) & (sc.shift(1) >= 38), -1, 0))
            pos = pd.Series(sig, index=tmp.index).replace(0, np.nan).ffill().shift().fillna(0)
            ret = tmp["Close"].pct_change().fillna(0)
            pnl = (pos * ret)[(pos * ret) != 0]
            gp = pnl[pnl > 0].sum(); gl = -pnl[pnl < 0].sum()
            pf = (gp/gl) if gl > 0 else 1.0
            if pf > best_pf:
                best_pf, best_p, best_df = pf, {"rsi":rp, "macd_f":mf, "macd_s":ms}, tmp
    return best_p, best_df

@dataclass
class StrategyMetrics:
    pf: float; win_rate: float; sharpe: float; max_dd: float; tot_ret: float; bh_ret: float; expectancy: float; adr: float

def backtest_strategy(df: pd.DataFrame) -> Tuple[StrategyMetrics, pd.Series]:
    sc, _ = conviction_score(df)
    sig = pd.Series(np.where((sc > 68) & (sc.shift(1) <= 68), 1, np.where((sc < 38) & (sc.shift(1) >= 38), -1, 0)), index=df.index)
    
    pos = sig.replace(0, np.nan).ffill().shift().fillna(0)
    pos = np.where(pos > 0, 1, 0)
    ret = df["Close"].pct_change().fillna(0)
    sret = pd.Series(pos, index=df.index) * ret
    eq = (1 + sret).cumprod()
    
    pnl = sret[sret != 0]
    wins = pnl[pnl > 0]; losses = pnl[pnl < 0]
    gp = wins.sum(); gl = -losses.sum()
    
    pf = (gp / gl) if gl > 0 else 99.0
    wr = len(wins) / len(pnl) if len(pnl) else 0.0
    sh = math.sqrt(252) * sret.mean() / sret.std() if sret.std() > 0 else 0.0
    dd = (eq / eq.cummax() - 1).min()
    
    avg_win = wins.mean() if len(wins) else 0
    avg_loss = abs(losses.mean()) if len(losses) else 0
    expectancy = (wr * avg_win) - ((1 - wr) * avg_loss)
    adr = (df["High"] - df["Low"]).mean() / df["Close"].mean() * 100
    
    m = StrategyMetrics(pf=pf, win_rate=wr, sharpe=sh, max_dd=dd, tot_ret=eq.iloc[-1]-1, bh_ret=(1+ret).cumprod().iloc[-1]-1, expectancy=expectancy, adr=adr)
    return m, eq

def risk_levels(df: pd.DataFrame) -> Dict[str, float]:
    last = df.iloc[-1]
    e = float(last["Close"]); a = float(last["ATR"])
    return {"entry": e, "stop": e - 2*a, "pt1": e + a, "pt2": e + 2*a, "pt3": e + 3*a, "atr": a}

# ============================================================
# Scanning Engine (With Fast Per-Stock Opt)
# ============================================================
@st.cache_data(ttl=15 * 60, show_spinner=False)
def run_scanner(tickers: List[str], scan_all: bool, max_n: int = 50) -> pd.DataFrame:
    rows = []
    to_scan = tickers if scan_all else tickers[:max_n]
    spy = fetch_ohlcv("SPY")
    spy_ret = (spy["Close"].iloc[-1] / spy["Close"].iloc[-126] - 1) * 100 if len(spy) > 126 else 0
    
    pb = st.progress(0, "Scanning universe & optimizing parameters...")
    for i, t in enumerate(to_scan):
        pb.progress((i + 1) / len(to_scan), text=f"Analyzing {t} ({i+1}/{len(to_scan)})")
        try:
            raw = fetch_ohlcv(t)
            if raw is None or len(raw) < 120: continue
            
            # Use only recent 1y for speed in scanner
            raw_trim = raw.tail(252)
            _, df = fast_optimize(raw_trim)
            sc, reason = conviction_score(df)
            
            last = df.iloc[-1]
            c6m = (last["Close"] / df["Close"].iloc[0] - 1) * 100
            
            rows.append({
                "Ticker": t,
                "Price": round(last["Close"], 2),
                "1D%": round((last["Close"]/df["Close"].iloc[-2]-1)*100, 2),
                "RS_vs_SPY": round(c6m - spy_ret, 2),
                "Score": round(sc.iloc[-1], 1),
                "RSI": round(last["RSI"], 1),
                "Verdict": "STRONG BUY" if sc.iloc[-1] >= 68 else "STRONG SELL" if sc.iloc[-1] <= 38 else "NEUTRAL",
                "Reason": reason,
                "Link": f"https://finance.yahoo.com/quote/{t}"
            })
        except: continue
    pb.empty()
    return pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)

# ============================================================
# Plotly Builders
# ============================================================
def build_main_chart(df: pd.DataFrame, ticker: str, view_period: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"))
    
    s = st.session_state
    if s["ov_ema20"]: fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA 20", line=dict(color="#f59e0b", width=1)))
    if s["ov_ema50"]: fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA 50", line=dict(color="#60a5fa", width=1)))
    if s["ov_ema200"]: fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], name="EMA 200", line=dict(color="#a78bfa", width=1.5, dash="dash")))
    if s["ov_avwap"]: fig.add_trace(go.Scatter(x=df.index, y=df["AVWAP"], name="1Y AVWAP", line=dict(color="#f87171", width=1.5, dash="dot")))
    if s["ov_super"]: fig.add_trace(go.Scatter(x=df.index, y=df["SUPER"], name="SuperTrend", line=dict(color="#34d399", width=1.5)))
    if s["ov_bb"]:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_UP"], name="BB Up", line=dict(color="rgba(148,163,184,0.3)", width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_LOW"], name="BB Low", line=dict(color="rgba(148,163,184,0.3)", width=1), fill="tonexty", fillcolor="rgba(148,163,184,0.05)"))
    if s["ov_ichi"]:
        fig.add_trace(go.Scatter(x=df.index, y=df["SENKOU_A"], name="Cloud A", line=dict(color="rgba(52,211,153,0.3)", width=0)))
        fig.add_trace(go.Scatter(x=df.index, y=df["SENKOU_B"], name="Cloud B", line=dict(color="rgba(248,113,113,0.3)", width=0), fill="tonexty", fillcolor="rgba(148,163,184,0.1)"))

    # Calculate default view window based on period selection
    end_dt = df.index[-1]
    days_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
    start_dt = end_dt - pd.Timedelta(days=days_map.get(view_period, 180))

    fig.update_layout(
        paper_bgcolor="#131722", plot_bgcolor="#131722",
        font=dict(color="#787b86", size=11),
        title=dict(text=f"<b>{ticker}</b> (Scroll to view history)", font=dict(color="#d1d4dc", size=14)),
        xaxis=dict(showgrid=True, gridcolor="#2a2e39", range=[start_dt, end_dt], rangeslider=dict(visible=False)),
        yaxis=dict(showgrid=True, gridcolor="#2a2e39", side="right", fixedrange=False),
        height=550, margin=dict(l=0, r=50, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
        dragmode="pan"
    )
    return fig

def build_subchart(df: pd.DataFrame, type_: str) -> go.Figure:
    fig = go.Figure()
    if type_ == "RSI":
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="#fbbf24", width=1.5)))
        fig.add_hline(y=70, line_dash="dash", line_color="#ef4444"); fig.add_hline(y=30, line_dash="dash", line_color="#10b981")
    elif type_ == "MACD":
        colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["MACD_HIST"]]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_HIST"], marker_color=colors))
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], line=dict(color="#60a5fa", width=1.2)))
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIG"], line=dict(color="#f87171", width=1.2)))
    elif type_ == "Volume":
        colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=colors))
        fig.add_trace(go.Scatter(x=df.index, y=df["VOL_SMA20"], line=dict(color="#f59e0b", width=1.5)))
    
    fig.update_layout(
        paper_bgcolor="#131722", plot_bgcolor="#131722", font=dict(color="#787b86", size=10),
        xaxis=dict(showgrid=True, gridcolor="#2a2e39"), yaxis=dict(showgrid=True, gridcolor="#2a2e39"),
        height=160, margin=dict(l=0, r=50, t=10, b=0), showlegend=False, dragmode="pan"
    )
    # Match the x-axis to the global setting roughly
    end_dt = df.index[-1]
    days_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
    start_dt = end_dt - pd.Timedelta(days=days_map.get(st.session_state["az_period"], 180))
    fig.update_xaxes(range=[start_dt, end_dt])
    return fig

# ============================================================
# Application UI
# ============================================================
_PCFG = {"scrollZoom": True, "displayModeBar": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

def main():
    init_settings()

    # Top Header
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:flex-end; padding-bottom:10px; border-bottom:1px solid #2a2e39; margin-bottom:15px;">
        <div style="display:flex; align-items:center; gap:12px;">
            <div style="background:#2962ff; width:38px; height:38px; border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:1.4rem;">📈</div>
            <div>
                <div style="font-size:1.2rem; font-weight:800; color:#d1d4dc;">TRADING TERMINAL PRO</div>
                <div style="font-size:0.7rem; color:#787b86; letter-spacing:0.1em;">SIMULATED EXECUTION · ADVANCED KPIs · DYNAMIC TABS</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar: Global Controls & Virtual Trading
    with st.sidebar:
        st.markdown("### 🔍 Global Symbol Sync")
        new_ticker = st.text_input("Add Ticker to Workspace", placeholder="e.g. TSLA").strip().upper()
        if new_ticker and new_ticker not in st.session_state["active_tabs"]:
            st.session_state["active_tabs"].append(new_ticker)
            save_tabs_state()
            st.rerun()
            
        st.markdown("### ⚙️ Timeframe")
        st.session_state["az_period"] = st.selectbox("Default Chart View", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2, help="Charts load 5yr data; this sets the default zoom.")

        st.markdown("---")
        st.markdown("### 💼 Virtual Trading")
        st.metric("Buying Power", f"${st.session_state['cash']:,.2f}")
        
        vt_tick = st.selectbox("Select Asset", st.session_state["active_tabs"]) if st.session_state["active_tabs"] else None
        vt_amt = st.number_input("Risk Amount ($)", min_value=10.0, max_value=st.session_state["cash"], value=min(500.0, st.session_state["cash"]), step=50.0)
        
        c1, c2 = st.columns(2)
        if c1.button("BUY", use_container_width=True) and vt_tick:
            raw = fetch_ohlcv(vt_tick)
            if not raw.empty:
                px_cur = raw["Close"].iloc[-1]
                shares = vt_amt / px_cur
                if st.session_state["cash"] >= vt_amt:
                    st.session_state["cash"] -= vt_amt
                    if vt_tick in st.session_state["portfolio"]:
                        old_s = st.session_state["portfolio"][vt_tick]["shares"]
                        old_p = st.session_state["portfolio"][vt_tick]["price"]
                        new_p = ((old_s * old_p) + vt_amt) / (old_s + shares)
                        st.session_state["portfolio"][vt_tick] = {"shares": old_s + shares, "price": new_p}
                    else:
                        st.session_state["portfolio"][vt_tick] = {"shares": shares, "price": px_cur}
                    st.session_state["trade_history"].append({"Time": datetime.now(), "Type": "BUY", "Asset": vt_tick, "Shares": shares, "Price": px_cur, "Value": vt_amt})
                    st.success(f"Bought {vt_tick}")
                    
        if c2.button("SELL ALL", use_container_width=True) and vt_tick:
            if vt_tick in st.session_state["portfolio"]:
                raw = fetch_ohlcv(vt_tick)
                if not raw.empty:
                    px_cur = raw["Close"].iloc[-1]
                    shares = st.session_state["portfolio"][vt_tick]["shares"]
                    val = shares * px_cur
                    st.session_state["cash"] += val
                    del st.session_state["portfolio"][vt_tick]
                    st.session_state["trade_history"].append({"Time": datetime.now(), "Type": "SELL", "Asset": vt_tick, "Shares": shares, "Price": px_cur, "Value": val})
                    st.success(f"Sold {vt_tick}")
        
        if st.session_state["portfolio"]:
            st.markdown("<div style='font-size:0.8rem; margin-top:10px; color:#787b86;'>Open Positions</div>", unsafe_allow_html=True)
            for k, v in st.session_state["portfolio"].items():
                st.markdown(f"<div style='font-size:0.85rem; color:#d1d4dc;'><b>{k}</b>: {v['shares']:.2f} sh @ ${v['price']:.2f}</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📐 Risk Management")
        acct_size = st.number_input("Account Size", value=5000.0)
        risk_pct = st.slider("Risk % per Trade", 0.5, 5.0, 1.0, 0.1)
        st.info(f"Recommended Risk Limit:\n**${(acct_size * risk_pct)/100:.2f}** per trade.")

    # Main Workspace Tabs
    tab_charts, tab_scan, tab_backtest, tab_news, tab_settings = st.tabs(["📈 CHARTS & ANALYSIS", "🔍 SCANNER & HEATMAP", "📊 BACKTEST & KPIs", "📰 NEWS & PULSE", "⚙️ SETTINGS"])

    # ════════════════════════════════════════════════════════
    # Tab 1: Dynamic Charts
    # ════════════════════════════════════════════════════════
    with tab_charts:
        if not st.session_state["active_tabs"]:
            st.info("Add a ticker symbol in the sidebar to start analyzing.")
        else:
            # Create sub-tabs for each active ticker
            chart_tabs = st.tabs(st.session_state["active_tabs"])
            for idx, tab in enumerate(chart_tabs):
                ticker = st.session_state["active_tabs"][idx]
                with tab:
                    c1, c2 = st.columns([10, 1])
                    if c2.button("✖ Close", key=f"close_{ticker}"):
                        st.session_state["active_tabs"].remove(ticker)
                        save_tabs_state()
                        st.rerun()

                    with st.spinner(f"Loading 5yr data & optimizing {ticker}..."):
                        raw = fetch_ohlcv(ticker)
                        if raw.empty:
                            st.error("No data found.")
                            continue
                        
                        # Optimize specifically for this ticker
                        best_p, df = fast_optimize(raw)
                        sc, reason = conviction_score(df)
                        rl = risk_levels(df)
                        
                        last = df.iloc[-1]
                        chg = (last["Close"]/df["Close"].iloc[-2]-1)*100
                        
                        # Top Metrics
                        m1, m2, m3, m4, m5, m6 = st.columns(6)
                        m1.metric("Price", f"${last['Close']:.2f}", f"{chg:+.2f}%")
                        m2.metric("Conviction", f"{sc.iloc[-1]:.0f}/100", "STRONG BUY" if sc.iloc[-1]>=68 else "SELL" if sc.iloc[-1]<=38 else "NEUTRAL")
                        m3.metric("RSI", f"{last['RSI']:.1f}")
                        m4.metric("MACD Hist", f"{last['MACD_HIST']:.3f}")
                        m5.metric("Opt. RSI Period", best_p["rsi"])
                        m6.metric("Opt. MACD", f"{best_p['macd_f']},{best_p['macd_s']}")

                        st.markdown(f"<div class='reason-box'><b>Signal Drivers:</b> {reason} (Indicators auto-tuned for {ticker})</div>", unsafe_allow_html=True)
                        
                        # Main Chart
                        st.plotly_chart(build_main_chart(df, ticker, st.session_state["az_period"]), use_container_width=True, config=_PCFG)
                        
                        # Sub-charts
                        sub_cols = st.columns(3)
                        if st.session_state["sc_rsi"]: sub_cols[0].plotly_chart(build_subchart(df, "RSI"), use_container_width=True, config=_PCFG)
                        if st.session_state["sc_macd"]: sub_cols[1].plotly_chart(build_subchart(df, "MACD"), use_container_width=True, config=_PCFG)
                        if st.session_state["sc_vol"]: sub_cols[2].plotly_chart(build_subchart(df, "Volume"), use_container_width=True, config=_PCFG)

                        # Trade Levels
                        with st.expander("Show Optimized Trade Levels & Stops", expanded=False):
                            pct_stop = abs(rl["entry"]-rl["stop"])/rl["entry"]*100
                            st.markdown(f"""
                            <div class="risk-card" style="margin-top:0;">
                                <div class="risk-row"><span class="risk-label">Entry</span><span style="font-weight:700">${rl['entry']:.2f}</span></div>
                                <div class="risk-row"><span class="risk-label">🛑 Dynamic ATR Stop</span><span class="risk-stop">${rl['stop']:.2f} <span style="font-size:0.7rem">(-{pct_stop:.1f}%)</span></span></div>
                                <div class="risk-row"><span class="risk-label">🎯 Target 1 (1R)</span><span class="risk-pt1">${rl['pt1']:.2f}</span></div>
                                <div class="risk-row"><span class="risk-label">🎯 Target 2 (2R)</span><span class="risk-pt2">${rl['pt2']:.2f}</span></div>
                            </div>
                            """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # Tab 2: Scanner & Heatmap
    # ════════════════════════════════════════════════════════
    with tab_scan:
        st.markdown("### 🔍 Advanced Universe Scanner")
        
        c1, c2, c3 = st.columns([2, 1, 1])
        universe_choice = c1.selectbox("Universe", ["Major ETFs & Funds", "S&P 500", "Nasdaq-100", "Dow Jones 30", "Custom List"], index=0)
        scan_all = c2.checkbox("Scan Entire List", value=False, help="Warning: Scanning 500 stocks takes ~1-2 mins.")
        max_scan = c3.number_input("Limit to N stocks", value=50, max_value=500, disabled=scan_all)
        
        if st.button("EXECUTE SCAN", type="primary", use_container_width=True):
            uni_list = get_universe(universe_choice)
            res = run_scanner(uni_list, scan_all, int(max_scan))
            
            if not res.empty:
                # Top Metrics
                buys = res[res["Verdict"] == "STRONG BUY"]
                sells = res[res["Verdict"] == "STRONG SELL"]
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Scanned", len(res))
                m2.metric("Strong Buys", len(buys))
                m3.metric("Strong Sells", len(sells))
                m4.metric("Bullish Ratio", f"{(len(buys)/len(res))*100:.1f}%" if len(res)>0 else "0%")

                # Heatmap
                st.markdown("#### 🗺️ Signal Heatmap")
                # Ensure sizes are positive for Treemap mapping
                res["Heatmap_Size"] = 1  
                fig_hm = px.treemap(
                    res, path=["Verdict", "Ticker"], values="Heatmap_Size",
                    color="Score", color_continuous_scale=["#ef4444", "#4b5563", "#10b981"],
                    range_color=[30, 75], title="Market Structure by Signal Strength"
                )
                fig_hm.update_layout(paper_bgcolor="#131722", plot_bgcolor="#131722", font=dict(color="#d1d4dc"), margin=dict(t=30, l=10, r=10, b=10))
                st.plotly_chart(fig_hm, use_container_width=True)

                # Results Table with clickable Links
                st.markdown("#### 📋 Actionable Setups")
                st.dataframe(
                    res.style.apply(lambda r: ["color:#10b981;font-weight:bold" if r["Verdict"]=="STRONG BUY" else "color:#ef4444;font-weight:bold" if r["Verdict"]=="STRONG SELL" else "" for _ in r], axis=1),
                    column_config={"Link": st.column_config.LinkColumn("Yahoo Finance", display_text="Open YF ↗")},
                    use_container_width=True, height=400
                )

    # ════════════════════════════════════════════════════════
    # Tab 3: Backtest & KPIs
    # ════════════════════════════════════════════════════════
    with tab_backtest:
        b1, b2 = st.columns([1, 3])
        with b1:
            st.markdown("### 📊 Engine Calibrator")
            bt_tick = st.selectbox("Select Ticker", st.session_state["active_tabs"]) if st.session_state["active_tabs"] else st.text_input("Ticker", "SPY")
            run_bt = st.button("RUN BACKTEST", type="primary", use_container_width=True)
            
            st.markdown("---")
            st.markdown("**What happens here?**")
            st.caption("The engine runs hundreds of indicator combinations on 5 years of historical data to find the setup that yields the highest Profit Factor for this specific asset, then simulates trading it.")

        with b2:
            if run_bt and bt_tick:
                with st.spinner("Running historical optimization grid..."):
                    raw = fetch_ohlcv(bt_tick)
                    if not raw.empty:
                        best_p, df_opt = fast_optimize(raw)
                        metrics, eq_curve = backtest_strategy(df_opt)
                        
                        # Display KPI Metrics Dashboard
                        st.markdown(f"### ⚡ Optimized Performance Profile: {bt_tick}")
                        st.markdown(f"<div style='color:#34d399; font-size:0.85rem; margin-bottom:15px;'>Winner Parameters: RSI {best_p['rsi']}, MACD {best_p['macd_f']}/{best_p['macd_s']}</div>", unsafe_allow_html=True)
                        
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Total Return", f"{metrics.tot_ret:.1%}", f"vs B&H {(metrics.tot_ret - metrics.bh_ret):+.1%}")
                        c2.metric("Win Rate", f"{metrics.win_rate:.1%}")
                        c3.metric("Profit Factor", f"{metrics.pf:.2f}", help="Gross Profit / Gross Loss")
                        c4.metric("Expectancy", f"{metrics.expectancy:.3f}% per trade", help="(Win% * Avg Win) - (Loss% * Avg Loss)")
                        
                        c5, c6, c7, c8 = st.columns(4)
                        c5.metric("Sharpe Ratio", f"{metrics.sharpe:.2f}")
                        c6.metric("Max Drawdown", f"{metrics.max_dd:.1%}")
                        c7.metric("Avg Daily Range (ADR)", f"{metrics.adr:.1f}%")
                        c8.metric("Market Regime", "Trend" if metrics.adr > 2.0 else "Chop")

                        # Equity Curve vs Buy & Hold
                        fig_eq = go.Figure()
                        # Base 100 normalization for pure comparison
                        s_eq_norm = eq_curve * 100
                        bh_eq_norm = (1 + df_opt["Close"].pct_change().fillna(0)).cumprod() * 100
                        
                        fig_eq.add_trace(go.Scatter(x=df_opt.index, y=s_eq_norm, name="Algo Strategy", line=dict(color="#2962ff", width=2.5)))
                        fig_eq.add_trace(go.Scatter(x=df_opt.index, y=bh_eq_norm, name="Buy & Hold", line=dict(color="#787b86", width=1.5, dash="dot")))
                        
                        fig_eq.update_layout(
                            paper_bgcolor="#131722", plot_bgcolor="#131722", font=dict(color="#d1d4dc"),
                            title="Capital Growth Comparison (Base 100)",
                            xaxis=dict(showgrid=True, gridcolor="#2a2e39"), yaxis=dict(showgrid=True, gridcolor="#2a2e39"),
                            height=350, margin=dict(l=0, r=0, t=40, b=0),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
                        )
                        st.plotly_chart(fig_eq, use_container_width=True)

    # ════════════════════════════════════════════════════════
    # Tab 4: News & Pulse
    # ════════════════════════════════════════════════════════
    with tab_news:
        st.markdown("### 📰 Market Pulse & Real-Time News")
        n1, n2 = st.columns([2, 1])
        
        with n1:
            st.markdown("#### Latest Headlines (SPY context)")
            news_items = fetch_news("SPY")
            if news_items:
                for item in news_items:
                    ts = item.get("providerPublishTime", time.time())
                    dt_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
                    link = item.get("link", "#")
                    title = item.get("title", "No Title")
                    pub = item.get("publisher", "Yahoo Finance")
                    
                    st.markdown(f"""
                    <a href="{link}" target="_blank" style="text-decoration:none;">
                        <div class="news-card">
                            <div class="news-title">{title}</div>
                            <div class="news-meta">{pub} • {dt_str}</div>
                        </div>
                    </a>
                    """, unsafe_allow_html=True)
            else:
                st.info("No news fetched. Check connection.")
                
        with n2:
            st.markdown("#### Broad Market Snapshot")
            with st.spinner("Fetching major indices..."):
                try:
                    majors = yf.download("SPY QQQ DIA IWM", period="5d", progress=False)["Close"]
                    if not majors.empty:
                        for c in ["SPY", "QQQ", "DIA", "IWM"]:
                            if c in majors.columns:
                                px_val = majors[c].iloc[-1]
                                chg = (px_val / majors[c].iloc[-2] - 1) * 100
                                st.metric(f"{c} ETF", f"${px_val:.2f}", f"{chg:+.2f}%")
                except:
                    st.caption("Unable to fetch indices right now.")

    # ════════════════════════════════════════════════════════
    # Tab 5: Settings & Overlays
    # ════════════════════════════════════════════════════════
    with tab_settings:
        s1, s2 = st.columns(2)
        with s1:
            st.markdown("#### Chart Overlays (Global)")
            st.session_state["ov_ema20"] = st.checkbox("EMA 20", st.session_state["ov_ema20"])
            st.session_state["ov_ema50"] = st.checkbox("EMA 50", st.session_state["ov_ema50"])
            st.session_state["ov_ema200"] = st.checkbox("EMA 200", st.session_state["ov_ema200"])
            st.session_state["ov_avwap"] = st.checkbox("Anchored VWAP (1y)", st.session_state["ov_avwap"])
            st.session_state["ov_super"] = st.checkbox("SuperTrend", st.session_state["ov_super"])
            st.session_state["ov_bb"] = st.checkbox("Bollinger Bands", st.session_state["ov_bb"])
            st.session_state["ov_ichi"] = st.checkbox("Ichimoku Cloud", st.session_state["ov_ichi"])
        
        with s2:
            st.markdown("#### Sub-charts (Global)")
            st.session_state["sc_rsi"] = st.checkbox("RSI", st.session_state["sc_rsi"])
            st.session_state["sc_macd"] = st.checkbox("MACD", st.session_state["sc_macd"])
            st.session_state["sc_vol"] = st.checkbox("Volume", st.session_state["sc_vol"])
            
            st.markdown("#### Custom Scanner List")
            c_raw = st.text_area("Custom Tickers (comma separated)", st.session_state["custom_tickers"])
            if c_raw != st.session_state["custom_tickers"]:
                st.session_state["custom_tickers"] = c_raw

if __name__ == "__main__":
    main()
