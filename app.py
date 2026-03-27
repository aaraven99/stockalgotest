import math
import os
import smtplib
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
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
    page_title="Trading Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── LocalStorage Custom Component Bridge ─────────────────────────────────────
@st.cache_resource
def get_ls_component():
    """Builds an invisible Javascript component to natively sync Python state to browser LocalStorage."""
    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
    <script>
      function sendMessageToStreamlitClient(type, data) {
        var outData = Object.assign({isStreamlitMessage: true, type: type}, data);
        window.parent.postMessage(outData, "*");
      }
      function init() {
        sendMessageToStreamlitClient("streamlit:componentReady", {apiVersion: 1});
      }
      function setFrameHeight() {
        sendMessageToStreamlitClient("streamlit:setFrameHeight", {height: 0});
      }
      window.addEventListener("message", function(event) {
        if (event.data.type !== "streamlit:render") return;
        var args = event.data.args;
        
        if (args.action === 'save') {
          if (window.lastSaveCounter === args.counter) return;
          window.lastSaveCounter = args.counter;
          try {
            window.localStorage.setItem(args.ls_key, JSON.stringify(args.data));
          } catch(e) {
            console.error("Save failed", e);
          }
        } else if (args.action === 'load') {
          if (window.hasLoaded) return;
          window.hasLoaded = true;
          try {
            var val = window.localStorage.getItem(args.ls_key);
            sendMessageToStreamlitClient("streamlit:setComponentValue", {value: {status: "loaded", data: val ? JSON.parse(val) : null}});
          } catch(e) {
            sendMessageToStreamlitClient("streamlit:setComponentValue", {value: {status: "error", error: e.message}});
          }
        }
      });
      init();
      setFrameHeight();
    </script>
    </head>
    <body></body>
    </html>
    """
    tmp_dir = tempfile.mkdtemp()
    with open(os.path.join(tmp_dir, "index.html"), "w") as f:
        f.write(html_code)
    return components.declare_component("local_storage_sync", path=tmp_dir)

ls_sync = get_ls_component()

# ── Settings initialization ──────────────────────────────────────────────────
PERSISTENT_KEYS = [
    "theme", "font", "scan_list", "custom_tickers", "auto_scan",
    "alert_browser", "alert_email", "alert_email_addr", "smtp_user", "smtp_pass",
    "scan_interval", "active_tickers", "az_period", "starting_capital",
    "paper_cash", "paper_portfolio", "paper_history",
    "layout_show_reasons", "layout_show_levels", "layout_show_kpis", "layout_show_sectors",
    "ov_ema20", "ov_ema50", "ov_ema200", "ov_avwap", "ov_bb", "ov_super",
    "ov_ichi", "ov_fib", "ov_psar", "sc_rsi", "sc_macd", "sc_vol",
    "sc_stoch", "sc_willr", "last_auto_scan"
]

def init_defaults():
    defaults = {
        "theme": "TradingView", "font": "JetBrains Mono", "scan_list": "S&P 500 + Nasdaq-100",
        "custom_tickers": "", "auto_scan": False, "alert_browser": False,
        "alert_email": False, "alert_email_addr": "", "smtp_user": "", "smtp_pass": "",
        "scan_interval": 15, "active_tickers": ["AAPL"], "az_period": "1y",
        "starting_capital": 5000.0, "paper_cash": 5000.0, "paper_portfolio": {},
        "paper_history": [], "layout_show_reasons": True, "layout_show_levels": True,
        "layout_show_kpis": True, "layout_show_sectors": True,
        "ov_ema20": True, "ov_ema50": True, "ov_ema200": True, "ov_avwap": True,
        "ov_bb": True, "ov_super": True, "ov_ichi": False, "ov_fib": False,
        "ov_psar": False, "sc_rsi": True, "sc_macd": True, "sc_vol": True,
        "sc_stoch": False, "sc_willr": False, "last_auto_scan": 0.0
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ── Global CSS (TradingView-inspired) ────────────────────────────────────────
_THEMES = {
    "Dark":             {"bg":"#0d1117","card":"#161b22","border":"#21262d","text":"#e6edf3","sub":"#8b949e","accent":"#3b82f6","chart":"#0d1117","grid":"#21262d"},
    "TradingView":      {"bg":"#131722","card":"#1e222d","border":"#2a2e39","text":"#d1d4dc","sub":"#787b86","accent":"#2962ff","chart":"#131722","grid":"#2a2e39"},
    "Dracula":          {"bg":"#282a36","card":"#21222c","border":"#44475a","text":"#f8f8f2","sub":"#6272a4","accent":"#bd93f9","chart":"#1e1f29","grid":"#383a59"},
}
_FONT_CSS = {
    "JetBrains Mono": ("@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');", "'JetBrains Mono', monospace"),
    "Inter":          ("@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');", "'Inter', sans-serif"),
    "System Sans":    ("", "'Segoe UI','Helvetica Neue',Arial,sans-serif"),
}

def inject_theme(theme: str, font: str):
    c = _THEMES.get(theme, _THEMES["TradingView"])
    font_import, ff = _FONT_CSS.get(font, _FONT_CSS["JetBrains Mono"])
    
    st.markdown(f"""<style>
    {font_import}
    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{ padding-top: 1rem !important; padding-bottom: 1rem !important; max-width: 100% !important; }}
    .stApp {{background-color:{c['bg']} !important;}}
    
    /* SAFE FONT OVERRIDE: Protects Material Icons from turning into text */
    p:not([class*="material"]), label, h1, h2, h3, h4, h5, h6, th, td, div.stMarkdown, [data-testid="stMetricValue"] {{font-family: {ff}, sans-serif !important;}}
    .stIcon, [class*="material"], .material-symbols-rounded, .material-icons {{font-family: 'Material Symbols Rounded', 'Material Icons' !important;}}
    
    .stTabs [data-baseweb="tab-list"] {{background-color:{c['card']};border-radius:8px;padding:4px;gap:4px;border:1px solid {c['border']};}}
    .stTabs [data-baseweb="tab"] {{color:{c['sub']};font-size:0.8rem;letter-spacing:0.05em;padding:6px 18px;border-radius:6px;}}
    .stTabs [aria-selected="true"] {{background-color:{c['bg']} !important;color:{c['accent']} !important;border-bottom:2px solid {c['accent']};}}
    
    [data-testid="stMetric"] {{background-color:{c['card']};border:1px solid {c['border']};border-radius:8px;padding:14px 18px;}}
    [data-testid="stMetricLabel"] {{color:{c['sub']} !important;font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;}}
    [data-testid="stMetricValue"] {{color:{c['text']} !important;font-size:1.3rem;font-weight:700;}}
    
    .verdict-card {{padding: 14px 24px; border-radius: 8px; font-size: 1.35rem; font-weight: 800; text-align: center; margin-bottom: 14px; font-family:{ff}, sans-serif !important;}}
    .verdict-buy     {{background: linear-gradient(135deg,#0d2e1e,#0d3d28); border: 1px solid #089981; color: #26a69a;}}
    .verdict-sell    {{background: linear-gradient(135deg,#2e0d0d,#3d1010); border: 1px solid #f23645; color: #ef5350;}}
    .verdict-neutral {{background: linear-gradient(135deg,#1a1d27,#1e222d); border: 1px solid #2a2e39; color: #787b86;}}

    .risk-card {{background-color:{c['card']}; border: 1px solid {c['border']}; border-radius: 8px; padding: 14px 18px; margin-top: 10px;}}
    .risk-row {{display: flex; justify-content: space-between; align-items: center; padding: 5px 0; border-bottom: 1px solid {c['border']};}}
    .risk-row:last-child {{border-bottom: none;}}
    .risk-label {{color: {c['sub']}; font-size: 0.78rem;}}
    .risk-stop  {{color: #f23645; font-weight: 700; font-size: 0.92rem;}}
    .risk-pt1   {{color: #f5a623; font-weight: 700; font-size: 0.92rem;}}
    .risk-pt2   {{color: #26a69a; font-weight: 700; font-size: 0.92rem;}}
    .risk-pt3   {{color: #8979ff; font-weight: 700; font-size: 0.92rem;}}
    
    .reason-box {{background-color:{c['card']}; border-left: 3px solid {c['accent']}; border-radius: 0 6px 6px 0; padding: 10px 14px; margin: 5px 0; font-size: 0.84rem; color: {c['text']};}}

    .pick-card {{background-color:{c['card']}; border: 1px solid {c['border']}; border-radius: 8px; padding: 14px 12px; text-align: center; transition: border-color 0.2s;}}
    .pick-card:hover {{border-color: {c['accent']};}}
    .pick-ticker a {{color: {c['accent']} !important; font-size: 1.1rem; font-weight: 800; text-decoration: none;}}
    .pick-score  {{color: #26a69a; font-size: 1.75rem; font-weight: 800;}}
    .pick-price  {{color: {c['text']}; font-size: 0.88rem; margin-top: 4px; font-weight: 600;}}
    .pick-chg-up   {{color: #26a69a; font-size: 0.82rem; font-weight: 600;}}
    .pick-chg-down {{color: #ef5350; font-size: 0.82rem; font-weight: 600;}}

    .news-card {{background-color:{c['card']}; border: 1px solid {c['border']}; border-radius: 8px; padding: 15px; margin-bottom: 12px; transition: border-color 0.2s, transform 0.2s; display: flex; gap: 15px; align-items: center;}}
    .news-card:hover {{border-color: {c['accent']}; transform: translateY(-2px);}}

    .close-tab-btn > div > div > button {{background-color: transparent !important; color: {c['sub']} !important; border: none !important; box-shadow: none !important; padding: 0 !important; font-size: 1.2rem !important; opacity: 0.3; transition: opacity 0.2s, color 0.2s;}}
    .close-tab-btn > div > div > button:hover {{color: #ef4444 !important; opacity: 1.0;}}

    .stButton>button {{background-color:{c['accent']} !important;color: #fff !important; border:none !important;border-radius:6px !important;font-family:{ff}, sans-serif !important;font-weight:600 !important;}}
    .stButton>button:hover {{background-color: #1e53e5 !important;}}
    div[data-baseweb="select"]>div, div[data-baseweb="input"]>div, .stTextInput input, .stNumberInput input, .stTextArea textarea {{background-color:{c['card']} !important;border-color:{c['border']} !important;color:{c['text']} !important;font-family:{ff}, sans-serif !important;}}
    </style>""", unsafe_allow_html=True)

# ── Terminal header ──────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:6px 0 14px 0;border-bottom:1px solid #2a2e39;margin-bottom:14px">
  <div style="display:flex;align-items:center;gap:14px">
    <div style="background:#2962ff;border-radius:6px;width:34px;height:34px;
                display:flex;align-items:center;justify-content:center;font-size:1.1rem">📈</div>
    <div>
      <div style="font-size:1.05rem;font-weight:800;color:#d1d4dc;letter-spacing:0.06em">TRADING TERMINAL PRO</div>
      <div style="font-size:0.65rem;color:#787b86;letter-spacing:0.14em;margin-top:1px">
        25+ INDICATORS · AUTO-CALIBRATED · SWING TRADE SIGNALS
      </div>
    </div>
  </div>
  <div style="font-size:0.68rem;color:#787b86;text-align:right">
    <span style="color:#26a69a">●</span> LIVE DATA &nbsp;·&nbsp; EOD PRICES
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Data / universe helpers
# ============================================================
DOW30 = sorted(["AAPL","AMGN","AXP","BA","CAT","CRM","CSCO","CVX","DIS","DOW","GS","HD","HON","IBM","JNJ","JPM","KO","MCD","MMM","MRK","MSFT","NKE","PG","TRV","UNH","V","VZ","WBA","WMT","INTC"])
ETFS_FUNDS = sorted(["SPY", "QQQ", "DIA", "IWM", "VTI", "VOO", "ARKK", "GLD", "SLV", "USO", "UNG", "TLT", "TMF", "XLF", "XLK", "XLE", "XLU", "XLV", "XLY", "XLP", "XLI", "XLB", "XLRE"])
FALLBACK_LIST = sorted(list(dict.fromkeys(DOW30 + ["NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "AVGO", "COST", "AMD", "NFLX", "ADBE", "QCOM", "AMAT", "INTU", "SBUX", "BKNG", "GILD"])))

@st.cache_data(ttl=60 * 60)
def fetch_universe() -> List[str]:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=headers, timeout=10)
        sp = pd.read_html(StringIO(r.text))[0]["Symbol"].astype(str).tolist()
    except: sp = []
    try:
        r = requests.get("https://en.wikipedia.org/wiki/Nasdaq-100", headers=headers, timeout=10)
        ndx = []
        for t in pd.read_html(StringIO(r.text)):
            cols = [str(c).strip().lower() for c in t.columns]
            col = next((c for c in cols if c in ["ticker", "symbol"]), None)
            if col: ndx = t[t.columns[cols.index(col)]].astype(str).tolist(); break
    except: ndx = []
    cleaned = [s.replace(".", "-").strip().upper() for s in sp + ndx if s]
    return sorted(list(dict.fromkeys(cleaned))) if cleaned else FALLBACK_LIST

@st.cache_data(ttl=60*60)
def fetch_sp500_only() -> List[str]:
    try:
        r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        return sorted([s.replace(".", "-").strip().upper() for s in pd.read_html(StringIO(r.text))[0]["Symbol"].astype(str).tolist()])
    except: return FALLBACK_LIST

@st.cache_data(ttl=60*60)
def fetch_ndx100_only() -> List[str]:
    try:
        r = requests.get("https://en.wikipedia.org/wiki/Nasdaq-100", headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        for tbl in pd.read_html(StringIO(r.text)):
            cols = [str(c).strip().lower() for c in tbl.columns]
            col = next((c for c in cols if c in ["ticker", "symbol"]), None)
            if col: return sorted([s.strip().upper() for s in tbl[tbl.columns[cols.index(col)]].astype(str).tolist()])
        return FALLBACK_LIST
    except: return FALLBACK_LIST

def get_universe(mode: str) -> List[str]:
    if mode == "Custom List":
        raw = st.session_state.get("custom_tickers", "")
        parsed = [t.strip().upper() for t in raw.replace(","," ").split() if t.strip()]
        return parsed if parsed else fetch_universe()
    if mode == "S&P 500": return fetch_sp500_only()
    if mode == "Nasdaq-100": return fetch_ndx100_only()
    if mode == "Dow Jones 30": return DOW30
    if mode == "Major ETFs & Funds": return ETFS_FUNDS
    return fetch_universe()

@st.cache_data(ttl=60*15)
def fetch_sector_performance() -> pd.DataFrame:
    sectors = {"Technology": "XLK", "Financials": "XLF", "Healthcare": "XLV", "Energy": "XLE", "Cons Discret": "XLY", "Industrials": "XLI"}
    try:
        df = yf.download(list(sectors.values()), period="5d", progress=False)["Close"]
        rows = []
        for name, ticker in sectors.items():
            if ticker in df.columns:
                px_cur = float(df[ticker].iloc[-1])
                chg_1d = (px_cur / float(df[ticker].iloc[-2]) - 1) * 100 if len(df) > 1 else 0
                chg_5d = (px_cur / float(df[ticker].iloc[0]) - 1) * 100 if len(df) > 1 else 0
                rows.append({"Sector": name, "ETF": ticker, "1D %": chg_1d, "5D %": chg_5d})
        return pd.DataFrame(rows).sort_values("1D %", ascending=False).reset_index(drop=True)
    except: return pd.DataFrame()

# ── Email alert helper & Push Notifications ──────────────────────────────────
def send_email_alert(to_addr: str, smtp_user: str, smtp_pass: str, subject: str, body: str) -> bool:
    try:
        msg = MIMEText(body, "plain")
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = to_addr
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as s:
            s.starttls(); s.login(smtp_user, smtp_pass); s.sendmail(smtp_user, to_addr, msg.as_string())
        return True
    except: return False

def push_browser_notification(title: str, body: str):
    components.html(f"""
    <script>
    (function(){{
        if (window.Notification && Notification.permission === 'granted') {{
            new Notification("{title}", {{body: "{body}", icon: "https://cdn-icons-png.flaticon.com/32/2168/2168252.png"}});
        }} else if (window.Notification && Notification.permission !== 'denied') {{
            Notification.requestPermission().then(p => {{
                if (p === 'granted') new Notification("{title}", {{body: "{body}", icon: "https://cdn-icons-png.flaticon.com/32/2168/2168252.png"}});
            }});
        }}
    }})();
    </script>
    """, height=0, width=0)

# ── Data Fetching ────────────────────────────────────────────────────────────
@st.cache_data(ttl=60 * 60)
def fetch_ohlcv(ticker: str, period: str = "5y") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        req_cols = ["Open", "High", "Low", "Close", "Volume"]
        for c in req_cols:
            if c not in df.columns: return pd.DataFrame()
        return df[req_cols].dropna()
    except: return pd.DataFrame()

@st.cache_data(ttl=60 * 15)
def fetch_market_news(ticker: str = "SPY") -> List[dict]:
    try: return yf.Ticker(ticker).news[:15]
    except: return []

# ============================================================
# Technical indicators & Strategy Base
# ============================================================
def ema(s: pd.Series, p: int) -> pd.Series: return s.ewm(span=p, adjust=False).mean()
def atr(df: pd.DataFrame, p: int = 14) -> pd.Series:
    tr = pd.concat([df["High"]-df["Low"], (df["High"]-df["Close"].shift()).abs(), (df["Low"]-df["Close"].shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(com=p-1, adjust=False).mean()

def add_all_indicators(df: pd.DataFrame, rsi_p: int=14, macd_f: int=12, macd_s: int=26) -> pd.DataFrame:
    x = df.copy()
    x["EMA20"] = ema(x["Close"], 20); x["EMA50"] = ema(x["Close"], 50); x["EMA200"] = ema(x["Close"], 200)
    
    delta = x["Close"].diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    ag = gain.ewm(com=rsi_p-1, adjust=False).mean(); al = loss.ewm(com=rsi_p-1, adjust=False).mean()
    x["RSI"] = 100 - (100 / (1 + ag / al.replace(0, np.nan)))
    
    ml = ema(x["Close"], macd_f) - ema(x["Close"], macd_s); sl = ema(ml, 9)
    x["MACD"] = ml; x["MACD_SIG"] = sl; x["MACD_HIST"] = ml - sl
    
    x["ATR"] = atr(x, 14); x["VOL_SMA20"] = x["Volume"].rolling(20).mean()
    x["AVWAP"] = ((x["High"]+x["Low"]+x["Close"])/3 * x["Volume"]).cumsum() / x["Volume"].cumsum()
    
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
    
    lo = x["Low"].rolling(14).min(); hi = x["High"].rolling(14).max()
    x["STO_K"] = 100 * ((x["Close"] - lo) / (hi - lo).replace(0, np.nan))
    x["STO_D"] = x["STO_K"].rolling(3).mean()
    x["WILLR"] = -100 * ((hi - x["Close"]) / (hi - lo).replace(0, np.nan))
    
    mid_bb = x["Close"].rolling(20).mean(); std_bb = x["Close"].rolling(20).std()
    x["BB_UP"] = mid_bb + 2*std_bb; x["BB_LOW"] = mid_bb - 2*std_bb; x["BB_MID"] = mid_bb
    
    x["TENKAN"] = (x["High"].rolling(9).max() + x["Low"].rolling(9).min()) / 2
    x["KIJUN"] = (x["High"].rolling(26).max() + x["Low"].rolling(26).min()) / 2
    x["SENKOU_A"] = ((x["TENKAN"] + x["KIJUN"]) / 2).shift(26)
    x["SENKOU_B"] = ((x["High"].rolling(52).max() + x["Low"].rolling(52).min()) / 2).shift(26)
    
    x["OBV"] = (np.sign(x["Close"].diff().fillna(0)) * x["Volume"]).cumsum()
    mfm = ((x["Close"]-x["Low"]) - (x["High"]-x["Close"])) / (x["High"]-x["Low"]).replace(0, np.nan)
    x["CMF"] = (mfm * x["Volume"]).rolling(20).sum() / x["Volume"].rolling(20).sum()
    
    return x

def conviction_score(df: pd.DataFrame) -> Tuple[pd.Series, List[str]]:
    score = pd.Series(50.0, index=df.index)
    score += np.where(df["Close"] > df["AVWAP"], 8, -8)
    score += np.where(df["OBV"] > df["OBV"].rolling(10).mean(), 4, -4)
    score += np.where(df["CMF"] > 0, 4, -4)
    score += np.where((df["Close"] > df["EMA20"]) & (df["EMA20"] > df["EMA50"]), 8, -8)
    score += np.where(df["Close"] > np.maximum(df["SENKOU_A"], df["SENKOU_B"]), 5, -5)
    score += np.where(df["Close"] > df["SUPER"], 5, -5)
    score += np.where((df["RSI"] > 50) & (df["RSI"] < 72), 6, -6)
    score += np.where(df["MACD_HIST"] > 0, 6, -6)
    score += np.where(df["STO_K"] > df["STO_D"], 3, -3)
    score += np.where(df["Close"] > df["BB_MID"], 3, -3)
    score += np.where(df["Volume"] > 1.4*df["VOL_SMA20"], 3, 0)
    score = score.clip(0, 100)
    
    last = df.iloc[-1]
    reasons = []
    if last["Close"] > last["AVWAP"]: reasons.append("Price is above Anchored VWAP (Support holding).")
    if last["EMA20"] > last["EMA50"]: reasons.append("Short-term EMA crossed above Mid-term EMA (Bullish trend).")
    if last["MACD_HIST"] > 0: reasons.append("MACD Histogram is positive (Momentum expanding).")
    if last["Close"] > last["SUPER"]: reasons.append("Price cleared SuperTrend resistance.")
    if not reasons: reasons.append("Consolidating market conditions.")
    return score, reasons[:3]

def build_signals(df: pd.DataFrame) -> pd.Series:
    s, _ = conviction_score(df)
    sig = pd.Series(0, index=df.index)
    sig[(s > 68) & (s.shift(1) <= 68)] = 1
    sig[(s < 38) & (s.shift(1) >= 38)] = -1
    return sig

@dataclass
class StrategyMetrics:
    profit_factor: float; win_rate: float; sharpe: float; max_drawdown: float; total_return: float; buy_hold_return: float; expectancy: float; adr: float

def backtest_strategy(df: pd.DataFrame, signal: pd.Series) -> StrategyMetrics:
    pos  = np.where(signal.replace(0, np.nan).ffill().shift().fillna(0) > 0, 1, 0)
    ret  = df["Close"].pct_change().fillna(0)
    sret = pd.Series(pos, index=df.index) * ret
    eq   = (1 + sret).cumprod()
    pnl  = sret[sret != 0]
    
    gp   = pnl[pnl > 0].sum(); gl = -pnl[pnl < 0].sum()
    pf   = float(gp / gl) if gl > 0 else 999.0
    wr   = float((pnl > 0).mean()) if len(pnl) else 0.0
    sh   = float(math.sqrt(252) * sret.mean() / sret.std()) if sret.std() > 0 else 0.0
    dd   = float((eq / eq.cummax() - 1).min()) if not eq.empty else 0.0
    tret = float(eq.iloc[-1] - 1) if not eq.empty else 0.0
    bh   = float((1 + ret).cumprod().iloc[-1] - 1) if not ret.empty else 0.0
    
    avg_win = float(pnl[pnl > 0].mean()) if len(pnl[pnl > 0]) else 0.0
    avg_loss = float(abs(pnl[pnl < 0].mean())) if len(pnl[pnl < 0]) else 0.0
    exp  = float((wr * avg_win) - ((1 - wr) * avg_loss))
    adr  = float(((df["High"] - df["Low"]) / df["Close"]).mean() * 100) if not df.empty else 0.0

    return StrategyMetrics(pf, wr, sh, dd, tret, bh, exp, adr)

@st.cache_data(ttl=60 * 60, show_spinner=False)
def optimize_params(raw: pd.DataFrame, fast: bool = False) -> Tuple[Dict[str, int], StrategyMetrics, pd.DataFrame]:
    best_obj, best_p, best_m, best_df = None, {"rsi": 14, "macd_fast": 12, "macd_slow": 26}, None, raw
    grid_rsi = [10, 14] if fast else [7, 9, 14]
    grid_macd = [(8,21), (12,26)] if fast else [(8,21), (12,26), (15,30)]
    
    for rp in grid_rsi:
        for f, s in grid_macd:
            tmp = add_all_indicators(raw, rp, f, s)
            sig = build_signals(tmp)
            m   = backtest_strategy(tmp, sig)
            obj = m.profit_factor * 0.6 + m.win_rate * 100 * 0.4
            if best_obj is None or obj > best_obj:
                best_obj, best_p, best_m, best_df = obj, {"rsi": rp, "macd_fast": f, "macd_slow": s}, m, tmp
    return best_p, best_m, best_df

def verdict_from_score(score: float) -> str:
    if score >= 72: return "STRONG BUY"
    if score <= 38: return "STRONG SELL"
    return "NEUTRAL"

def risk_levels(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty: return {"entry": 1.0, "stop": 1.0, "pt1": 1.0, "pt2": 1.0, "pt3": 1.0, "atr": 0.0}
    last  = df.iloc[-1]
    e = float(last["Close"]); a = float(last.get("ATR", 0.0))
    return {"entry": e, "stop": e - 2*a, "pt1": e + a, "pt2": e + 2*a, "pt3": e + 3*a, "atr": a}


# ============================================================
# Chart builders
# ============================================================
def build_candlestick_chart(df: pd.DataFrame, ticker: str, view_period: str, s: dict) -> go.Figure:
    tc = _THEMES.get(st.session_state.get("theme", "TradingView"), _THEMES["TradingView"])
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name=ticker, increasing_line_color="#26a69a", decreasing_line_color="#ef5350"))
    if s.get("ov_ema20"): fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA 20", line=dict(color="#f59e0b", width=1)))
    if s.get("ov_ema50"): fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA 50", line=dict(color="#60a5fa", width=1)))
    if s.get("ov_ema200"): fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], name="EMA 200", line=dict(color="#a78bfa", width=1, dash="dash")))
    if s.get("ov_avwap"): fig.add_trace(go.Scatter(x=df.index, y=df["AVWAP"], name="AVWAP", line=dict(color="#f87171", width=1.5, dash="dashdot")))
    if s.get("ov_bb"):
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_UP"], name="BB Up", line=dict(color="rgba(148,163,184,0.4)", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_LOW"], name="BB Low", line=dict(color="rgba(148,163,184,0.4)", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(148,163,184,0.04)"))
    if s.get("ov_super"): fig.add_trace(go.Scatter(x=df.index, y=df["SUPER"], name="SuperTrend", line=dict(color="#34d399", width=1.5)))
    
    end_dt = df.index[-1] if not df.empty else pd.Timestamp.now()
    days_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
    start_dt = end_dt - pd.Timedelta(days=days_map.get(view_period, 180))

    fig.update_layout(
        paper_bgcolor=tc["chart"], plot_bgcolor=tc["chart"], font=dict(color=tc["sub"], size=11),
        title=dict(text=f"<b>{ticker}</b>", font=dict(color=tc["text"], size=15), pad=dict(b=8)),
        xaxis=dict(showgrid=True, gridcolor=tc["grid"], zeroline=False, rangeslider=dict(visible=False), color=tc["sub"], range=[start_dt, end_dt]),
        yaxis=dict(showgrid=True, gridcolor=tc["grid"], zeroline=False, color=tc["sub"], side="right"),
        height=520, dragmode="pan", margin=dict(l=0, r=60, t=60, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
    )
    return fig

def build_score_chart(df, score, signal, view_period) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=score, name="Score", fill="tozeroy", fillcolor="rgba(59,130,246,0.12)", line=dict(color="#3b82f6",width=2)))
    fig.add_hline(y=72, line_dash="dash", line_color="#10b981", annotation_text="BUY")
    fig.add_hline(y=38, line_dash="dash", line_color="#ef4444", annotation_text="SELL")
    buys  = df.index[signal == 1]; sells = df.index[signal == -1]
    if len(buys): fig.add_trace(go.Scatter(x=buys, y=score[buys], mode="markers", marker=dict(symbol="triangle-up", color="#10b981", size=11)))
    if len(sells): fig.add_trace(go.Scatter(x=sells, y=score[sells], mode="markers", marker=dict(symbol="triangle-down", color="#ef4444", size=11)))
    
    end_dt = df.index[-1] if not df.empty else pd.Timestamp.now()
    start_dt = end_dt - pd.Timedelta(days={"1mo":30,"3mo":90,"6mo":180,"1y":365,"2y":730,"5y":1825}.get(view_period, 180))

    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font=dict(family="monospace", color="#8b949e"),
        title=dict(text="Conviction Score", font=dict(color="#e6edf3",size=13)),
        xaxis=dict(showgrid=True, gridcolor="#21262d", range=[start_dt, end_dt]),
        yaxis=dict(showgrid=True, gridcolor="#21262d", range=[0,100]),
        height=220, dragmode="pan", margin=dict(l=0,r=0,t=40,b=0), showlegend=False
    )
    return fig

def build_sub_chart(df, view_period, type_: str) -> go.Figure:
    fig = go.Figure()
    if type_ == "Volume":
        colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(df["Close"], df["Open"])]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=colors))
        fig.add_trace(go.Scatter(x=df.index, y=df["VOL_SMA20"], line=dict(color="#f59e0b",width=1.5)))
    elif type_ == "RSI":
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], line=dict(color="#fbbf24",width=2)))
        fig.add_hline(y=70, line_dash="dash", line_color="#ef4444"); fig.add_hline(y=30, line_dash="dash", line_color="#10b981")
        fig.update_yaxes(range=[0, 100])
    elif type_ == "MACD":
        colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["MACD_HIST"]]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_HIST"], marker_color=colors))
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], line=dict(color="#60a5fa",width=1.5)))
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIG"], line=dict(color="#f87171",width=1.5)))
    
    end_dt = df.index[-1] if not df.empty else pd.Timestamp.now()
    start_dt = end_dt - pd.Timedelta(days={"1mo":30,"3mo":90,"6mo":180,"1y":365,"2y":730,"5y":1825}.get(view_period, 180))

    fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font=dict(family="monospace",color="#8b949e"),
                       title=dict(text=type_, font=dict(color="#e6edf3",size=12)),
                       xaxis=dict(showgrid=True, gridcolor="#21262d", range=[start_dt, end_dt]),
                       yaxis=dict(showgrid=True, gridcolor="#21262d"),
                       height=160, showlegend=False, dragmode="pan", margin=dict(l=0,r=0,t=35,b=0))
    return fig

def build_pnl_scatter(df, signal) -> go.Figure:
    pos  = np.where(signal.replace(0, np.nan).ffill().shift().fillna(0) > 0, 1, 0)
    ret  = df["Close"].pct_change().fillna(0)
    sret = pd.Series(pos, index=df.index) * ret
    pnl = sret[sret != 0] * 100
    colors = ["#10b981" if val > 0 else "#ef4444" for val in pnl]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pnl.index, y=pnl, mode='markers', marker=dict(size=8, color=colors, line=dict(width=1, color='#131722'))))
    fig.add_hline(y=0, line_dash="dash", line_color="#4b5563")
    fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", font=dict(family="monospace", color="#8b949e"),
        title=dict(text="Trade PnL %", font=dict(color="#e6edf3", size=13)),
        xaxis=dict(showgrid=True, gridcolor="#21262d"), yaxis=dict(showgrid=True, gridcolor="#21262d"),
        height=250, margin=dict(l=0, r=0, t=40, b=0), showlegend=False
    )
    return fig


# ============================================================
# Market scanner
# ============================================================
@st.cache_data(ttl=30 * 60, show_spinner=False)
def scan_universe(tickers: List[str], max_scan: int, auto_tune: bool = False) -> pd.DataFrame:
    rows = []
    spy_raw = fetch_ohlcv("SPY", "1y")
    spy_ret = float((spy_raw["Close"].iloc[-1] / spy_raw["Close"].iloc[0] - 1) * 100) if spy_raw is not None and not spy_raw.empty else 0.0

    pb = st.progress(0, "Scanning universe...")
    target = tickers[:max_scan]
    for i, t in enumerate(target):
        pb.progress((i + 1) / len(target), text=f"Analyzing & Auto-Tuning {t} ({i+1}/{len(target)})")
        try:
            raw = fetch_ohlcv(t, "1y") 
            if raw.empty or len(raw) < 120: continue
            if auto_tune:
                bp, _, _ = optimize_params(raw, fast=True)
                df = add_all_indicators(raw, bp["rsi"], bp["macd_fast"], bp["macd_slow"])
            else: df = add_all_indicators(raw, 14, 12, 26)
            
            sc, _ = conviction_score(df)
            last_c = float(df["Close"].iloc[-1])
            chg_1d = float((last_c / float(df["Close"].iloc[-2]) - 1) * 100)
            chg_6m = float((last_c / float(df["Close"].iloc[int(len(df)/2)]) - 1) * 100)
            
            rows.append({
                "Ticker": t, "Price": round(last_c, 2), "1D Chg%": round(chg_1d, 2),
                "6M Chg%": round(chg_6m, 2), "RS vs SPY": round(chg_6m - spy_ret, 2),
                "Score": round(float(sc.iloc[-1]), 1), "RSI": round(float(df["RSI"].iloc[-1]), 1),
                "Stop Loss": round(last_c - 2*float(df["ATR"].iloc[-1]), 2),
                "Verdict": verdict_from_score(float(sc.iloc[-1]))
            })
        except: continue
    pb.empty()
    return pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)


def _run_analysis(t: str):
    raw = fetch_ohlcv(t, "5y")
    if raw is None or raw.empty: return None, None, None, None, None, None, None
    if len(raw) >= 120: bp, _, df = optimize_params(raw)
    else: bp = {"rsi": 14, "macd_fast": 12, "macd_slow": 26}; df = add_all_indicators(raw, 14, 12, 26)
    score, reasons = conviction_score(df)
    return df, score, build_signals(df), reasons, risk_levels(df), bp, raw


def main():
    if "ls_loaded" not in st.session_state:
        init_defaults()
        res = ls_sync(action="load", ls_key="trading_term_v6", key="ls_loader")
        if res is None: st.spinner("Syncing workspace..."); st.stop()
        else:
            if res.get("status") == "loaded" and res.get("data"):
                for k, v in res["data"].items():
                    if k in ["starting_capital", "paper_cash"]:
                        try: st.session_state[k] = float(v)
                        except: st.session_state[k] = 5000.0
                    elif k in ["scan_interval"]:
                        try: st.session_state[k] = int(v)
                        except: st.session_state[k] = 15
                    else: st.session_state[k] = v
            st.session_state["ls_loaded"] = True; st.rerun()

    inject_theme(st.session_state["theme"], st.session_state["font"])
    universe = get_universe(st.session_state["scan_list"])
    
    # Check Auto Scan
    if st.session_state.get("auto_scan", False):
        interval = int(st.session_state.get("scan_interval", 15)) * 60
        now = time.time(); last = float(st.session_state.get("last_auto_scan", 0.0))
        if now - last >= interval:
            st.session_state["last_auto_scan"] = now
            with st.spinner(f"Auto-scan running on all {len(universe)} stocks…"):
                res = scan_universe(tuple(universe), len(universe), auto_tune=True)
            if not res.empty:
                st.session_state["last_scan_results"] = res
                buys = res[res["Verdict"] == "STRONG BUY"]
                top5 = buys.head(5) if len(buys) >= 1 else res.head(5)
                if not top5.empty:
                    title = f"Trading Alert — Top {len(top5)} Picks Detected"
                    body = f"Trading Terminal Auto-Scan completed at {datetime.now().strftime('%Y-%m-%d %H:%M')}.\n\nYour Top Picks:\n\n"
                    for _, row in top5.iterrows(): body += f"• {row['Ticker']}: {row['Verdict']} (Score: {row['Score']}/100) | Price: ${row['Price']:.2f}\n"
                    if st.session_state.get("alert_browser"):
                        st.session_state["pending_browser_notif"] = {"title": title, "body": f"{top5.iloc[0]['Ticker']} score {top5.iloc[0]['Score']}/100"}
                    if st.session_state.get("alert_email") and st.session_state.get("alert_email_addr"):
                        if send_email_alert(st.session_state["alert_email_addr"], st.session_state.get("smtp_user", ""), st.session_state.get("smtp_pass", ""), title, body):
                            st.toast("✅ Auto-scan email successfully sent!")
                        else: st.toast("⚠️ Auto-scan email failed. Check App Password.", icon="⚠️")
                    else: st.toast("✅ Auto-scan completed!", icon="✅")

    with st.sidebar:
        if st.session_state.get("layout_show_sectors", True):
            st.markdown("### 🔄 Sector Rotation Watchlist")
            sdf = fetch_sector_performance()
            if not sdf.empty:
                for _, r in sdf.iterrows(): st.metric(f"{r['Sector']} ({r['ETF']})", f"{r['1D %']:+.2f}%", f"{r['5D %']:+.2f}% (5D)")

    tabs = st.tabs(["📈 ANALYZE", "🔍 SCANNER", "📊 BACKTEST", "💼 PAPER TRADING", "📰 NEWS", "⚙️ SETTINGS"])

    # ════════════════════════════════════════════════════════
    # TAB 1: ANALYZE
    # ════════════════════════════════════════════════════════
    with tabs[0]:
        c1, c2 = st.columns([1, 3], gap="medium")
        with c1:
            with st.expander("Workspace & Timeframe", expanded=True):
                t_in = st.text_input("Add Ticker", placeholder="e.g. AAPL").strip().upper()
                if t_in and t_in not in st.session_state.active_tickers:
                    st.session_state.active_tickers.append(t_in); st.rerun()
                st.session_state["az_period"] = st.selectbox("Default Zoom", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
            with st.expander("Overlays", expanded=False):
                st.session_state["ov_ema20"] = st.checkbox("EMA 20", st.session_state["ov_ema20"])
                st.session_state["ov_ema50"] = st.checkbox("EMA 50", st.session_state["ov_ema50"])
                st.session_state["ov_ema200"] = st.checkbox("EMA 200", st.session_state["ov_ema200"])
                st.session_state["ov_avwap"] = st.checkbox("AVWAP", st.session_state["ov_avwap"])
                st.session_state["ov_bb"] = st.checkbox("Bollinger", st.session_state["ov_bb"])
                st.session_state["ov_super"] = st.checkbox("SuperTrend", st.session_state["ov_super"])
            with st.expander("Sub-charts", expanded=False):
                st.session_state["sc_rsi"] = st.checkbox("RSI", st.session_state["sc_rsi"])
                st.session_state["sc_macd"] = st.checkbox("MACD", st.session_state["sc_macd"])
                st.session_state["sc_vol"] = st.checkbox("Volume", st.session_state["sc_vol"])

        with c2:
            if not st.session_state.active_tickers: st.info("Add a ticker on the left.")
            else:
                w_tabs = st.tabs(st.session_state.active_tickers)
                for idx, t in enumerate(st.session_state.active_tickers):
                    with w_tabs[idx]:
                        col_x1, col_x2 = st.columns([15, 1])
                        with col_x2:
                            st.markdown('<div class="close-tab-btn">', unsafe_allow_html=True)
                            if st.button("✖", key=f"cl_{t}"): st.session_state.active_tickers.remove(t); st.rerun()
                            st.markdown('</div>', unsafe_allow_html=True)

                        with st.spinner(f"Analyzing {t}..."):
                            df, score, sig, reasons, rl, bp, _ = _run_analysis(t)

                        if df is not None:
                            sc_val = float(score.iloc[-1])
                            verd = verdict_from_score(sc_val)
                            cls_m = {"STRONG BUY":"verdict-buy","STRONG SELL":"verdict-sell","NEUTRAL":"verdict-neutral"}
                            
                            st.markdown(f'<div class="verdict-card {cls_m.get(verd,"verdict-neutral")}"><a href="https://finance.yahoo.com/quote/{t}" target="_blank" style="color:inherit;text-decoration:none;">{t} ↗</a> &nbsp;·&nbsp; {verd} &nbsp;·&nbsp; Score {sc_val:.0f}/100</div>', unsafe_allow_html=True)

                            last = df.iloc[-1]
                            chg = (float(last["Close"])/float(df["Close"].iloc[-2])-1)*100 if len(df)>1 else 0.0
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Price", f"${last['Close']:.2f}", f"{chg:+.2f}%")
                            m2.metric("RSI", f"{last['RSI']:.1f}")
                            m3.metric("MACD Hist", f"{last['MACD_HIST']:.2f}")
                            m4.metric("ATR", f"${last['ATR']:.2f}")

                            if st.session_state.get("layout_show_reasons"):
                                st.markdown('<div style="font-size:0.68rem;letter-spacing:0.12em;color:#8b949e;margin:20px 0 6px 0">SIGNAL DRIVERS</div>', unsafe_allow_html=True)
                                for i, r in enumerate(reasons, 1): st.markdown(f'<div class="reason-box">{i}. {r}</div>', unsafe_allow_html=True)

                            if st.session_state.get("layout_show_levels"):
                                ep = max(float(rl.get("entry", 1.0)), 0.0001)
                                pct_s = abs(rl["entry"]-rl["stop"]) / ep * 100
                                pct_1 = (rl["pt1"]-rl["entry"]) / ep * 100
                                pct_2 = (rl["pt2"]-rl["entry"]) / ep * 100
                                pct_3 = (rl["pt3"]-rl["entry"]) / ep * 100
                                st.markdown(f"""
                                <div class="risk-card">
                                  <div class="risk-row"><span class="risk-label">Entry</span><b>${rl['entry']:.2f}</b></div>
                                  <div class="risk-row"><span class="risk-label">🛑 Stop</span><span class="risk-stop">${rl['stop']:.2f} <span style="font-size:0.75rem">(-{pct_s:.1f}%)</span></span></div>
                                  <div class="risk-row"><span class="risk-label">🎯 Target 1</span><span class="risk-pt1">${rl['pt1']:.2f} <span style="font-size:0.75rem">(+{pct_1:.1f}%)</span></span></div>
                                  <div class="risk-row"><span class="risk-label">🎯 Target 2</span><span class="risk-pt2">${rl['pt2']:.2f} <span style="font-size:0.75rem">(+{pct_2:.1f}%)</span></span></div>
                                  <div class="risk-row"><span class="risk-label">🎯 Target 3</span><span class="risk-pt3">${rl['pt3']:.2f} <span style="font-size:0.75rem">(+{pct_3:.1f}%)</span></span></div>
                                </div>""", unsafe_allow_html=True)

                            st.markdown('<div style="margin-top:16px"></div>', unsafe_allow_html=True)
                            
                            c_fig = build_candlestick_chart(df, t, st.session_state["az_period"], st.session_state)
                            b_px = df.index[sig == 1]; s_px = df.index[sig == -1]
                            if len(b_px): c_fig.add_trace(go.Scatter(x=b_px, y=df["Low"][b_px]*0.99, mode="markers", marker=dict(symbol="triangle-up", color="#10b981", size=11), name="BUY"))
                            if len(s_px): c_fig.add_trace(go.Scatter(x=s_px, y=df["High"][s_px]*1.01, mode="markers", marker=dict(symbol="triangle-down", color="#ef4444", size=11), name="SELL"))
                            st.plotly_chart(c_fig, use_container_width=True, config=_PCFG)
                            st.plotly_chart(build_score_chart(df, score, sig, st.session_state["az_period"]), use_container_width=True, config=_PCFG)

                            if st.session_state["sc_vol"]: st.plotly_chart(build_sub_chart(df, st.session_state["az_period"], "Volume"), use_container_width=True, config=_PCFG)
                            if st.session_state["sc_rsi"]: st.plotly_chart(build_sub_chart(df, st.session_state["az_period"], "RSI"), use_container_width=True, config=_PCFG)
                            if st.session_state["sc_macd"]: st.plotly_chart(build_sub_chart(df, st.session_state["az_period"], "MACD"), use_container_width=True, config=_PCFG)

                            # Ticker News
                            st.markdown(f'<div style="margin-top:20px; font-size:0.7rem; color:#8b949e;">LATEST NEWS FOR {t}</div>', unsafe_allow_html=True)
                            for n in fetch_market_news(t)[:3]:
                                dt = datetime.fromtimestamp(n.get("providerPublishTime", time.time())).strftime("%Y-%m-%d")
                                st.markdown(f'• <a href="{n.get("link","#")}" target="_blank">{n.get("title")}</a> ({dt})', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # TAB 2: SCANNER
    # ════════════════════════════════════════════════════════
    with tabs[1]:
        sc1, sc2, sc3 = st.columns([2, 1, 1], gap="medium")
        with sc1:
            scan_all = st.checkbox("Scan Entire Universe", value=True)
            limit_val = st.slider("Limit (if not all)", 20, 200, 80, disabled=scan_all)
            auto_t = st.checkbox("Auto-Tune Params", value=True, help="Re-tunes optimization for every single stock using its past 1 year of data.")
        with sc3:
            st.markdown('<div style="margin-top:26px"></div>', unsafe_allow_html=True)
            if st.button("SCAN NOW", type="primary", use_container_width=True):
                n = len(universe) if scan_all else limit_val
                with st.spinner(f"Scanning {n} stocks..."):
                    st.session_state["last_scan_results"] = scan_universe(tuple(universe), n, auto_t)

        if "last_scan_results" in st.session_state and not st.session_state["last_scan_results"].empty:
            res = st.session_state["last_scan_results"]
            b = res[res["Verdict"] == "STRONG BUY"]
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Scanned", len(res))
            c2.metric("Strong Buys", len(b))
            c3.metric("Bearish", len(res[res["Verdict"] == "STRONG SELL"]))

            st.markdown('<div style="font-size:0.68rem;letter-spacing:0.12em;color:#10b981;margin:22px 0 10px 0">TOP PICKS</div>', unsafe_allow_html=True)
            top5 = b.head(5) if not b.empty else res.head(5)
            cols = st.columns(min(len(top5), 5))
            for col, (_, r) in zip(cols, top5.iterrows()):
                c_cls = "pick-chg-up" if r["1D Chg%"] >= 0 else "pick-chg-down"
                col.markdown(f"""
                <div class="pick-card">
                  <div class="pick-ticker"><a href="https://finance.yahoo.com/quote/{r['Ticker']}" target="_blank">{r['Ticker']} ↗</a></div>
                  <div class="pick-score">{r['Score']}</div>
                  <div class="{c_cls}">${r['Price']:.2f} ({r['1D Chg%']:+.2f}%)</div>
                </div>""", unsafe_allow_html=True)

            st.markdown('<div style="margin:20px 0 10px 0;font-size:0.7rem;color:#8b949e;">SIGNAL HEATMAP</div>', unsafe_allow_html=True)
            res["Sz"] = 1
            fig_hm = px.treemap(res, path=["Verdict", "Ticker"], values="Sz", color="Score", color_continuous_scale=["#ef4444", "#4b5563", "#10b981"], range_color=[30, 75])
            fig_hm.update_layout(margin=dict(t=0,l=0,r=0,b=0), paper_bgcolor="#131722")
            st.plotly_chart(fig_hm, use_container_width=True)

            st.dataframe(res.drop(columns=["Sz"], errors="ignore"), column_config={"YF Link": st.column_config.LinkColumn("YF")}, use_container_width=True, height=400)

    # ════════════════════════════════════════════════════════
    # TAB 3: BACKTEST
    # ════════════════════════════════════════════════════════
    with tabs[2]:
        bx, by = st.columns([1, 2], gap="medium")
        with bx:
            bt_t = st.text_input("Ticker", "AAPL").upper()
            bt_p = st.selectbox("Window", ["1y", "2y", "5y"], index=1)
            start_cap = float(st.session_state.get("starting_capital", 5000.0))
            cap_in = st.number_input("Starting Capital ($)", min_value=100.0, value=start_cap, step=100.0, format="%.2f")
            st.session_state["starting_capital"] = float(cap_in)
            run_bt = st.button("RUN BACKTEST", type="primary", use_container_width=True)

        with by:
            if run_bt and bt_t:
                with st.spinner(f"Backtesting {bt_t}..."):
                    raw = fetch_ohlcv(bt_t, bt_p)
                    if not raw.empty:
                        bp, m, df_bt = optimize_params(raw)
                        sig = build_signals(df_bt)
                        pos = np.where(sig.replace(0, np.nan).ffill().shift().fillna(0) > 0, 1, 0)
                        ret = df_bt["Close"].pct_change().fillna(0)
                        eq = (1 + pd.Series(pos, index=df_bt.index) * ret).cumprod() * cap_in
                        bh = (1 + ret).cumprod() * cap_in
                        
                        st.markdown(f"#### Results: {bt_t}")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Strategy Final", f"${float(eq.iloc[-1]):,.2f}", f"{m.total_return:+.1%}")
                        c2.metric("Buy & Hold", f"${float(bh.iloc[-1]):,.2f}", f"{m.buy_hold_return:+.1%}")
                        c3.metric("Win Rate", f"{m.win_rate:.1%}")
                        c4.metric("Profit Factor", f"{m.profit_factor:.2f}")

                        c5, c6, c7, c8 = st.columns(4)
                        c5.metric("Sharpe", f"{m.sharpe:.2f}")
                        c6.metric("Max DD", f"{m.max_drawdown:.1%}")
                        c7.metric("Expectancy", f"{m.expectancy:.2%}")
                        c8.metric("ADR", f"{m.adr:.2f}%")

                        st.plotly_chart(build_pnl_scatter(df_bt, sig), use_container_width=True, config=_PCFG)
                        
                        fig_eq = go.Figure()
                        fig_eq.add_trace(go.Scatter(x=df_bt.index, y=eq, name="Strategy", line=dict(color="#2962ff", width=2)))
                        fig_eq.add_trace(go.Scatter(x=df_bt.index, y=bh, name="B&H", line=dict(color="#787b86", dash="dot")))
                        fig_eq.update_layout(paper_bgcolor="#131722", plot_bgcolor="#131722", font=dict(color="#d1d4dc"), margin=dict(t=30, b=0, l=0, r=0))
                        st.plotly_chart(fig_eq, use_container_width=True, config=_PCFG)

    # ════════════════════════════════════════════════════════
    # TAB 4: PAPER TRADING
    # ════════════════════════════════════════════════════════
    with tabs[3]:
        p1, p2 = st.columns([1, 2], gap="large")
        with p1:
            st.markdown("### 💼 Execute")
            pc_val = float(st.session_state.get("paper_cash", 5000.0))
            st.metric("Virtual Balance", f"${pc_val:,.2f}")
            
            pt_t = st.selectbox("Asset", st.session_state.active_tickers) if st.session_state.active_tickers else None
            safe_max = max(10.0, pc_val)
            pt_amt = st.number_input("Amount ($)", min_value=10.0, max_value=safe_max, value=min(500.0, safe_max), step=50.0, format="%.2f")
            
            c_b, c_s = st.columns(2)
            if c_b.button("BUY", type="primary", use_container_width=True) and pt_t:
                raw = fetch_ohlcv(pt_t, "1mo")
                if not raw.empty:
                    px = float(raw["Close"].iloc[-1])
                    sh = float(pt_amt) / px
                    if pc_val >= float(pt_amt):
                        st.session_state["paper_cash"] = pc_val - float(pt_amt)
                        port = st.session_state["paper_portfolio"]
                        if pt_t in port:
                            o_sh = port[pt_t]["shares"]; o_px = port[pt_t]["avg_price"]
                            port[pt_t]["avg_price"] = ((o_sh * o_px) + float(pt_amt)) / (o_sh + sh)
                            port[pt_t]["shares"] += sh
                        else: port[pt_t] = {"shares": sh, "avg_price": px}
                        st.session_state["paper_history"].append({"Time": datetime.now().strftime("%Y-%m-%d %H:%M"), "Action": "BUY", "Ticker": pt_t, "Price": px, "Shares": sh})
                        st.success("Bought!")
                        st.rerun()
            
            if c_s.button("SELL ALL", use_container_width=True) and pt_t:
                port = st.session_state["paper_portfolio"]
                if pt_t in port:
                    raw = fetch_ohlcv(pt_t, "1mo")
                    if not raw.empty:
                        px = float(raw["Close"].iloc[-1])
                        sh = port[pt_t]["shares"]
                        val = sh * px
                        st.session_state["paper_cash"] += val
                        del port[pt_t]
                        st.session_state["paper_history"].append({"Time": datetime.now().strftime("%Y-%m-%d %H:%M"), "Action": "SELL", "Ticker": pt_t, "Price": px, "Shares": sh})
                        st.success("Sold!")
                        st.rerun()

            st.markdown("---")
            st.markdown("### 📐 Risk Manager")
            r_acc = float(st.number_input("Account Size", value=pc_val, step=100.0, format="%.2f"))
            r_pct = float(st.slider("Risk Limit %", 0.5, 5.0, 1.0, 0.1))
            r_sl = float(st.number_input("Stop Loss %", value=5.0, step=0.5, format="%.1f"))
            max_r = (r_acc * r_pct) / 100.0
            max_p = max_r / (r_sl / 100.0) if r_sl > 0 else 0.0
            st.info(f"Max Risk: **${max_r:.2f}** | Max Pos: **${max_p:.2f}**")

        with p2:
            st.markdown("### 🗃️ Open Positions")
            if st.session_state["paper_portfolio"]:
                rows = []
                for k, v in st.session_state["paper_portfolio"].items():
                    raw = fetch_ohlcv(k, "1mo")
                    px = float(raw["Close"].iloc[-1]) if not raw.empty else v["avg_price"]
                    pnl = (px / v["avg_price"] - 1) * 100
                    rows.append({"Ticker": k, "Shares": round(v["shares"],4), "Entry": f"${v['avg_price']:.2f}", "Current": f"${px:.2f}", "Value": f"${v['shares']*px:.2f}", "PnL": f"{pnl:+.2f}%"})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else: st.caption("No open positions.")
            
            st.markdown("### 📜 History")
            if st.session_state["paper_history"]: st.dataframe(pd.DataFrame(st.session_state["paper_history"]).tail(10), use_container_width=True, hide_index=True)
            else: st.caption("No history.")

    # ════════════════════════════════════════════════════════
    # TAB 5: NEWS
    # ════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown("### 📰 Market Pulse")
        n1, n2 = st.columns([3, 1], gap="large")
        with n1:
            nt = st.text_input("Ticker search:", "SPY").upper()
            items = fetch_market_news(nt)
            if items:
                for i in items:
                    dt = datetime.fromtimestamp(i.get("providerPublishTime", time.time())).strftime("%Y-%m-%d %H:%M")
                    img = ""
                    try: img = i["thumbnail"]["resolutions"][0]["url"]
                    except: pass
                    img_html = f'<img src="{img}" style="width:80px;height:80px;border-radius:6px;object-fit:cover;">' if img else '<div style="width:80px;height:80px;background:#2a2e39;border-radius:6px;"></div>'
                    st.markdown(f"""
                    <div style="display:flex;gap:15px;margin-bottom:15px;background:#1e222d;padding:15px;border-radius:8px;">
                      {img_html}
                      <div>
                        <a href="{i.get('link','#')}" target="_blank" style="color:#2962ff;font-size:1.1rem;font-weight:bold;text-decoration:none;">{i.get('title')}</a>
                        <div style="color:#787b86;font-size:0.8rem;margin-top:5px;">{i.get('publisher')} • {dt}</div>
                      </div>
                    </div>""", unsafe_allow_html=True)
            else: st.info("No news found.")
        with n2:
            for c in ["SPY", "QQQ", "DIA", "BTC-USD"]:
                raw = fetch_ohlcv(c, "1mo")
                if not raw.empty and len(raw) > 1:
                    st.metric(c, f"${raw['Close'].iloc[-1]:.2f}", f"{(raw['Close'].iloc[-1]/raw['Close'].iloc[-2]-1)*100:+.2f}%")

    # ════════════════════════════════════════════════════════
    # TAB 6: SETTINGS
    # ════════════════════════════════════════════════════════
    with tabs[5]:
        s1, s2 = st.columns(2, gap="large")
        with s1:
            if st.selectbox("Theme", list(_THEMES.keys()), index=list(_THEMES.keys()).index(st.session_state["theme"])) != st.session_state["theme"]:
                st.session_state["theme"] = _THEMES.keys()[0]; sync_settings(); st.rerun()
            if st.selectbox("Font", list(_FONT_CSS.keys()), index=list(_FONT_CSS.keys()).index(st.session_state["font"])) != st.session_state["font"]:
                st.session_state["font"] = _FONT_CSS.keys()[0]; sync_settings(); st.rerun()
            st.session_state["layout_show_reasons"] = st.checkbox("Show Logic Drivers", st.session_state["layout_show_reasons"])
            st.session_state["layout_show_levels"] = st.checkbox("Show Trade Levels", st.session_state["layout_show_levels"])
            st.session_state["layout_show_kpis"] = st.checkbox("Show KPIs", st.session_state["layout_show_kpis"])
            st.session_state["layout_show_sectors"] = st.checkbox("Show Sidebar Sectors", st.session_state["layout_show_sectors"])
        with s2:
            st.session_state["scan_list"] = st.selectbox("Universe", ["Major ETFs & Funds", "S&P 500 + Nasdaq-100", "S&P 500", "Nasdaq-100", "Dow Jones 30", "Custom List"], index=0)
            if st.session_state["scan_list"] == "Custom List":
                st.session_state["custom_tickers"] = st.text_area("Custom Tickers", st.session_state["custom_tickers"])
            
            st.session_state["auto_scan"] = st.toggle("Enable auto-scan", st.session_state["auto_scan"])
            if st.session_state["auto_scan"]:
                st.session_state["scan_interval"] = st.select_slider("Interval", [5,10,15,30,60], st.session_state["scan_interval"])
                st.session_state["alert_browser"] = st.checkbox("Browser notifications", st.session_state["alert_browser"])
                if st.session_state["alert_browser"]:
                    components.html("<script>Notification.requestPermission();</script>", height=0)
                st.session_state["alert_email"] = st.checkbox("Email alerts", st.session_state["alert_email"])
                if st.session_state["alert_email"]:
                    st.session_state["alert_email_addr"] = st.text_input("Send to", st.session_state["alert_email_addr"])
                    st.session_state["smtp_user"] = st.text_input("Gmail", st.session_state["smtp_user"])
                    st.session_state["smtp_pass"] = st.text_input("App Pass", st.session_state["smtp_pass"], type="password")

    if "pending_browser_notif" in st.session_state:
        n = st.session_state.pop("pending_browser_notif")
        push_browser_notification(n["title"], n["body"])

    st.session_state.save_counter = st.session_state.get("save_counter", 0) + 1
    ls_sync(action="save", ls_key="trading_term_v6", data={k: st.session_state[k] for k in PERSISTENT_KEYS if k in st.session_state}, counter=st.session_state.save_counter, key="ls_saver")

if __name__ == "__main__":
    main()
