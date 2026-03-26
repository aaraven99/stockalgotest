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
            // Silent save to prevent infinite rerun loops
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


# ── Global CSS (TradingView-inspired) ────────────────────────────────────────
st.markdown("""
<style>
  #MainMenu, footer, header { visibility: hidden; }

  /* ── Layout ── */
  .block-container { padding-top: 1rem !important; padding-bottom: 1rem !important; max-width: 100% !important; }
  .stApp { background-color: #131722; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background-color: #1e222d; border: 1px solid #2a2e39;
    border-radius: 8px; padding: 4px; gap: 2px; margin-bottom: 8px;
  }
  .stTabs [data-baseweb="tab"] {
    color: #787b86; font-size: 0.82rem; letter-spacing: 0.06em;
    padding: 7px 20px; border-radius: 6px; border: none !important;
  }
  .stTabs [aria-selected="true"] {
    background-color: #131722 !important;
    color: #2962ff !important;
    border-bottom: 2px solid #2962ff !important;
  }

  /* ── Metric cards ── */
  [data-testid="stMetric"] {
    background-color: #1e222d; border: 1px solid #2a2e39;
    border-radius: 6px; padding: 12px 14px;
  }
  [data-testid="stMetricLabel"] { color: #787b86 !important; font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase; }
  [data-testid="stMetricValue"] { color: #d1d4dc !important; font-size: 1.2rem; font-weight: 700; }
  [data-testid="stMetricDelta"] { font-size: 0.78rem; }

  /* ── Verdict banner ── */
  .verdict-card {
    padding: 14px 24px; border-radius: 8px;
    font-size: 1.35rem; font-weight: 800; text-align: center; margin-bottom: 14px; letter-spacing: 0.04em;
  }
  .verdict-buy     { background: linear-gradient(135deg,#0d2e1e,#0d3d28); border: 1px solid #089981; color: #26a69a; }
  .verdict-sell    { background: linear-gradient(135deg,#2e0d0d,#3d1010); border: 1px solid #f23645; color: #ef5350; }
  .verdict-neutral { background: linear-gradient(135deg,#1a1d27,#1e222d); border: 1px solid #2a2e39; color: #787b86; }

  /* ── Risk card ── */
  .risk-card {
    background-color: #1e222d; border: 1px solid #2a2e39;
    border-radius: 8px; padding: 14px 18px; margin-top: 10px;
  }
  .risk-row { display: flex; justify-content: space-between; align-items: center; padding: 5px 0; border-bottom: 1px solid #2a2e39; }
  .risk-row:last-child { border-bottom: none; }
  .risk-label { color: #787b86; font-size: 0.78rem; }
  .risk-stop  { color: #f23645; font-weight: 700; font-size: 0.92rem; }
  .risk-pt1   { color: #f5a623; font-weight: 700; font-size: 0.92rem; }
  .risk-pt2   { color: #26a69a; font-weight: 700; font-size: 0.92rem; }
  .risk-pt3   { color: #8979ff; font-weight: 700; font-size: 0.92rem; }

  /* ── Reason box ── */
  .reason-box {
    background-color: #1e222d; border-left: 3px solid #2962ff;
    border-radius: 0 6px 6px 0; padding: 10px 14px; margin: 5px 0;
    font-size: 0.84rem; color: #d1d4dc;
  }

  /* ── Scanner pick cards ── */
  .pick-card {
    background-color: #1e222d; border: 1px solid #2a2e39;
    border-radius: 8px; padding: 14px 12px; text-align: center;
    transition: border-color 0.2s;
  }
  .pick-card:hover { border-color: #2962ff; }
  .pick-ticker { color: #2962ff; font-size: 1.1rem; font-weight: 800; }
  .pick-score  { color: #26a69a; font-size: 1.75rem; font-weight: 800; }
  .pick-price  { color: #d1d4dc; font-size: 0.88rem; margin-top: 4px; font-weight: 600; }
  .pick-chg-up   { color: #26a69a; font-size: 0.82rem; font-weight: 600; }
  .pick-chg-down { color: #ef5350; font-size: 0.82rem; font-weight: 600; }

  /* ── Chrome-like Close Tab Buttons ── */
  .close-tab-btn > div > div > button {
      background-color: transparent !important;
      color: #787b86 !important;
      border: none !important;
      box-shadow: none !important;
      padding: 0 !important;
      font-size: 1.2rem !important;
      opacity: 0.3;
      transition: opacity 0.2s, color 0.2s;
  }
  .close-tab-btn > div > div > button:hover {
      color: #ef4444 !important;
      opacity: 1.0;
  }

  /* ── News Card ── */
  .news-card {
    background-color: #1e222d; border: 1px solid #2a2e39;
    border-radius: 8px; padding: 15px; margin-bottom: 12px;
    transition: border-color 0.2s, transform 0.2s;
  }
  .news-card:hover {
    border-color: #2962ff;
    transform: translateY(-2px);
  }

  /* ── Buttons ── */
  .stButton > button {
    background-color: #2962ff !important; color: #fff !important;
    border: none !important; border-radius: 6px !important;
    font-weight: 700 !important; letter-spacing: 0.05em; padding: 8px 20px !important;
  }
  .stButton > button:hover { background-color: #1e53e5 !important; }

  /* ── Inputs & selects ── */
  div[data-baseweb="select"] > div { background-color: #1e222d !important; border-color: #2a2e39 !important; color: #d1d4dc !important; }
  div[data-baseweb="input"]  > div { background-color: #1e222d !important; border-color: #2a2e39 !important; }
  .stTextInput input, .stNumberInput input, .stTextArea textarea {
    background-color: #1e222d !important; border-color: #2a2e39 !important;
    color: #d1d4dc !important; border-radius: 6px !important;
  }

  /* ── Expanders ── */
  .streamlit-expanderHeader {
    background-color: #1e222d !important; border-radius: 6px !important;
    font-size: 0.82rem; color: #787b86 !important; border-color: #2a2e39 !important;
  }

  /* ── Checkboxes & labels ── */
  .stCheckbox label { color: #b2b5be !important; font-size: 0.82rem; }
  label { color: #787b86 !important; font-size: 0.8rem; }

  /* ── Slider ── */
  .stSlider [data-testid="stSlider"] > div > div > div { background-color: #2962ff !important; }

  /* ── Dataframe ── */
  .stDataFrame { border-radius: 8px; overflow: hidden; border: 1px solid #2a2e39; }

  /* ── Dividers ── */
  hr { border-color: #2a2e39; margin: 14px 0; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #1e222d; }
  ::-webkit-scrollbar-thumb { background: #2a2e39; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Terminal header ──────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:6px 0 14px 0;border-bottom:1px solid #2a2e39;margin-bottom:14px">
  <div style="display:flex;align-items:center;gap:14px">
    <div style="background:#2962ff;border-radius:6px;width:34px;height:34px;
                display:flex;align-items:center;justify-content:center;font-size:1.1rem">📈</div>
    <div>
      <div style="font-size:1.05rem;font-weight:800;color:#d1d4dc;letter-spacing:0.06em">TRADING TERMINAL</div>
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
@st.cache_data(ttl=60 * 60)
def fetch_universe() -> List[str]:
    fallback = sorted(list(dict.fromkeys([
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "AVGO", "COST",
        "AMD", "NFLX", "ADBE", "QCOM", "CSCO", "INTC", "AMGN", "TXN", "PEP", "CMCSA",
        "TMUS", "HON", "AMAT", "INTU", "SBUX", "BKNG", "GILD", "ADP", "LRCX", "MU",
        "PANW", "MELI", "ISRG", "VRTX", "KLAC", "SNPS", "CDNS", "CRWD", "MRVL", "ASML",
        "JPM", "V", "MA", "BRK-B", "UNH", "XOM", "JNJ", "WMT", "PG", "LLY", "CVX", "HD",
        "ABBV", "MRK", "KO", "BAC", "ORCL", "CRM", "MCD", "NKE", "PFE", "TMO", "ABT",
        "DHR", "WFC", "LIN", "ACN", "DIS", "VZ", "CMG", "COP", "NEE", "PM", "LOW", "UNP",
        "GS", "RTX", "C", "SPGI", "BLK", "CAT", "DE", "BA", "UPS", "T", "IBM", "NOW",
        "MS", "SCHW", "PLTR", "UBER", "SHOP", "SQ", "PYPL", "RIVN", "F", "GM",
    ])))
    headers = {"User-Agent": "Mozilla/5.0 (compatible; TradingTerminal/2.0)"}

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
        if not ndx:
            raise ValueError("no ndx")
        cleaned = [s.replace(".", "-").strip().upper() for s in sp + ndx if s]
        return sorted(list(dict.fromkeys(cleaned)))
    except Exception:
        return fallback


DOW30 = sorted([
    "AAPL","AMGN","AXP","BA","CAT","CRM","CSCO","CVX","DIS","DOW",
    "GS","HD","HON","IBM","JNJ","JPM","KO","MCD","MMM","MRK",
    "MSFT","NKE","PG","TRV","UNH","V","VZ","WBA","WMT","INTC",
])

# Major ETFs & Funds
ETFS_FUNDS = sorted([
    "SPY", "QQQ", "DIA", "IWM", "VTI", "VOO", "ARKK", "GLD", "SLV", "USO", 
    "UNG", "TLT", "TMF", "XLF", "XLK", "XLE", "XLU", "XLV", "XLY", "XLP", "XLI", "XLB", "XLRE"
])

@st.cache_data(ttl=60*60)
def fetch_sp500_only() -> List[str]:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; TradingTerminal/2.0)"}
    try:
        r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=headers, timeout=20)
        tbl = pd.read_html(StringIO(r.text))[0]
        return sorted([s.replace(".", "-").strip().upper() for s in tbl["Symbol"].astype(str).tolist()])
    except Exception:
        return fetch_universe()

@st.cache_data(ttl=60*60)
def fetch_ndx100_only() -> List[str]:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; TradingTerminal/2.0)"}
    try:
        r = requests.get("https://en.wikipedia.org/wiki/Nasdaq-100", headers=headers, timeout=20)
        for tbl in pd.read_html(StringIO(r.text)):
            cols = [str(c).strip().lower() for c in tbl.columns]
            if "ticker" in cols:
                return sorted([s.strip().upper() for s in tbl[tbl.columns[cols.index("ticker")]].astype(str).tolist()])
    except Exception:
        pass
    return fetch_universe()

def get_universe(mode: str) -> List[str]:
    if mode == "Custom List":
        raw = st.session_state.get("custom_tickers", "")
        parsed = [t.strip().upper() for t in raw.replace(","," ").split() if t.strip()]
        return parsed if parsed else fetch_universe()
    if mode == "S&P 500":
        return fetch_sp500_only() or fetch_universe()
    elif mode == "Nasdaq-100":
        return fetch_ndx100_only() or fetch_universe()
    elif mode == "Dow Jones 30":
        return DOW30
    elif mode == "Major ETFs & Funds":
        return ETFS_FUNDS
    return fetch_universe()

@st.cache_data(ttl=60*15)
def fetch_sector_performance() -> pd.DataFrame:
    sectors = {
        "Technology": "XLK", "Financials": "XLF", "Healthcare": "XLV",
        "Energy": "XLE", "Cons Discret": "XLY", "Industrials": "XLI"
    }
    try:
        df = yf.download(list(sectors.values()), period="5d", progress=False)["Close"]
        rows = []
        for name, ticker in sectors.items():
            if ticker in df.columns:
                px_cur = df[ticker].iloc[-1]
                chg_1d = (px_cur / df[ticker].iloc[-2] - 1) * 100 if len(df) > 1 else 0
                chg_5d = (px_cur / df[ticker].iloc[0] - 1) * 100 if len(df) > 1 else 0
                rows.append({"Sector": name, "ETF": ticker, "1D %": chg_1d, "5D %": chg_5d})
        return pd.DataFrame(rows).sort_values("1D %", ascending=False).reset_index(drop=True)
    except: return pd.DataFrame()


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
        "sc_stoch": False, "sc_willr": False,
        "save_counter": 0, "last_auto_scan": 0.0
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Theme CSS injection ───────────────────────────────────────────────────────
_THEMES = {
    "Dark":             {"bg":"#0d1117","card":"#161b22","border":"#21262d","text":"#e6edf3","sub":"#8b949e","accent":"#3b82f6","chart":"#0d1117","grid":"#21262d"},
    "TradingView":      {"bg":"#131722","card":"#1e222d","border":"#2a2e39","text":"#d1d4dc","sub":"#787b86","accent":"#2962ff","chart":"#131722","grid":"#2a2e39"},
    "Dracula":          {"bg":"#282a36","card":"#21222c","border":"#44475a","text":"#f8f8f2","sub":"#6272a4","accent":"#bd93f9","chart":"#1e1f29","grid":"#383a59"},
    "Nord":             {"bg":"#2e3440","card":"#3b4252","border":"#434c5e","text":"#eceff4","sub":"#7b88a1","accent":"#88c0d0","chart":"#272c36","grid":"#3b4252"},
    "Monokai":          {"bg":"#272822","card":"#1d1e1a","border":"#3e3d32","text":"#f8f8f2","sub":"#75715e","accent":"#a6e22e","chart":"#1d1e1a","grid":"#3e3d32"},
    "Cyberpunk":        {"bg":"#0d0d0d","card":"#1a1a2e","border":"#16213e","text":"#00ff9f","sub":"#00b4d8","accent":"#0ff","chart":"#050510","grid":"#0a0a2a"},
    "Midnight Blue":    {"bg":"#07090f","card":"#0d1121","border":"#1a2035","text":"#e2e8f0","sub":"#64748b","accent":"#6366f1","chart":"#07090f","grid":"#1a2035"},
    "Bloomberg":        {"bg":"#0a0a0a","card":"#111111","border":"#2a1e00","text":"#ff8c00","sub":"#996600","accent":"#ff8c00","chart":"#080800","grid":"#1f1800"},
    "Solarized Dark":   {"bg":"#002b36","card":"#073642","border":"#124952","text":"#fdf6e3","sub":"#657b83","accent":"#268bd2","chart":"#002b36","grid":"#073642"},
    "Rose Pine":        {"bg":"#191724","card":"#1f1d2e","border":"#26233a","text":"#e0def4","sub":"#6e6a86","accent":"#c4a7e7","chart":"#191724","grid":"#26233a"},
}

_FONT_CSS = {
    "JetBrains Mono": ("@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');", "'JetBrains Mono', monospace"),
    "IBM Plex Mono":  ("@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;700&display=swap');", "'IBM Plex Mono', monospace"),
    "Source Code Pro": ("@import url('https://fonts.googleapis.com/css2?family=Source+Code+Pro:wght@400;700&display=swap');", "'Source Code Pro', monospace"),
    "Inter":          ("@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');", "'Inter', sans-serif"),
    "Roboto":         ("@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');", "'Roboto', sans-serif"),
    "Roboto Mono":    ("@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&display=swap');", "'Roboto Mono', monospace"),
    "System Mono":    ("", "monospace"),
    "System Sans":    ("", "'Segoe UI','Helvetica Neue',Arial,sans-serif"),
}

def inject_theme(theme: str, font: str):
    c = _THEMES.get(theme, _THEMES["Dark"])
    font_import, ff = _FONT_CSS.get(font, _FONT_CSS["JetBrains Mono"])
    
    # SAFE FONT OVERRIDE logic: strictly apply fonts to our text, avoid polluting Material Symbols
    st.markdown(f"""<style>
    {font_import}
    .stApp {{background-color:{c['bg']} !important;}}
    
    /* Apply font to headers and standard text, EXCLUDING specific icon classes */
    p:not([class*="material"]), 
    label, h1, h2, h3, h4, h5, h6, 
    th, td, div.stMarkdown, 
    [data-testid="stMetricValue"] {{font-family: {ff}, sans-serif !important;}}
    
    /* Explicitly PROTECT material icons from font overrides to fix overlapping bug */
    .stIcon, [class*="material"], .material-symbols-rounded, .material-icons {{
        font-family: 'Material Symbols Rounded', 'Material Icons' !important;
    }}
    
    [data-testid="stMetric"] {{background-color:{c['card']};border:1px solid {c['border']};border-radius:8px;padding:14px 18px;}}
    [data-testid="stMetricLabel"] {{color:{c['sub']} !important;font-size:0.72rem;letter-spacing:0.08em;text-transform:uppercase;}}
    [data-testid="stMetricValue"] {{color:{c['text']} !important;font-size:1.3rem;font-weight:700;}}
    [data-testid="stMetricDelta"] {{font-size:0.78rem;}}
    .stTabs [data-baseweb="tab-list"] {{background-color:{c['card']};border-radius:8px;padding:4px;gap:4px;border:1px solid {c['border']};}}
    .stTabs [data-baseweb="tab"] {{color:{c['sub']};font-size:0.8rem;letter-spacing:0.05em;padding:6px 18px;border-radius:6px;}}
    .stTabs [aria-selected="true"] {{background-color:{c['bg']} !important;color:{c['accent']} !important;border-bottom:2px solid {c['accent']};}}
    .verdict-card {{font-family:{ff}, sans-serif !important;}}
    .risk-card, .reason-box, .pick-card {{background-color:{c['card']};border-color:{c['border']};}}
    div[data-baseweb="select"]>div {{background-color:{c['card']} !important;border-color:{c['border']} !important;color:{c['text']} !important;}}
    div[data-baseweb="input"]>div {{background-color:{c['card']} !important;border-color:{c['border']} !important;}}
    .stButton>button {{background-color:{c['accent']} !important;color: #fff !important; border:none !important;border-radius:6px !important;font-family:{ff}, sans-serif !important;font-weight:600 !important;}}
    .stButton>button:hover {{background-color: #1e53e5 !important;}}
    .stTextInput input, .stNumberInput input, .stTextArea textarea {{background-color:{c['card']} !important;border-color:{c['border']} !important;color:{c['text']} !important;font-family:{ff}, sans-serif !important;}}
    .streamlit-expanderHeader {{background-color:{c['card']} !important;border-color:{c['border']} !important;color:{c['sub']} !important;border-radius:6px !important;}}
    .stCheckbox label {{color:{c['sub']} !important;}}
    </style>""", unsafe_allow_html=True)


# ── Email alert helper ────────────────────────────────────────────────────────
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
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False


# ── Browser notification JS injector ─────────────────────────────────────────
def push_browser_notification(title: str, body: str):
    components.html(f"""
    <script>
    (function(){{
        if (window.Notification && Notification.permission === 'granted') {{
            var notification = new Notification("{title}", {{
                body: "{body}",
                icon: "https://cdn-icons-png.flaticon.com/32/2168/2168252.png"
            }});
        }} else if (window.Notification && Notification.permission !== 'denied') {{
            Notification.requestPermission().then(function (permission) {{
                if (permission === 'granted') {{
                    var notification = new Notification("{title}", {{
                        body: "{body}",
                        icon: "https://cdn-icons-png.flaticon.com/32/2168/2168252.png"
                    }});
                }}
            }});
        }}
    }})();
    </script>
    """, height=1, width=1)


# ── Auto-scan logic ───────────────────────────────────────────────────────────
def check_auto_scan(universe: List[str]):
    if not st.session_state.get("auto_scan", False):
        return
    interval = st.session_state.get("scan_interval", 15) * 60
    now = time.time()
    last = st.session_state.get("last_auto_scan", 0.0)
    
    if now - last < interval:
        return
        
    st.session_state["last_auto_scan"] = now
    
    with st.spinner(f"Auto-scan running on all {len(universe)} stocks…"):
        results = scan_universe(tuple(universe), len(universe), auto_tune=False)
        
    if results.empty:
        return
        
    # Save the background scan to memory so the Scanner Tab can display it
    st.session_state["last_scan_results"] = results
        
    buys = results[results["Verdict"] == "STRONG BUY"]
    top5 = buys.head(5) if len(buys) >= 1 else results.head(5)
    
    if top5.empty:
        return
        
    top = top5.iloc[0]
    
    st.session_state["auto_top_ticker"] = top["Ticker"]
    st.session_state["auto_top_score"]  = top["Score"]
    
    # Format comprehensive Email with Top 5 Picks
    title = f"Trading Alert — Top {len(top5)} Picks Detected"
    
    body = f"Trading Terminal Auto-Scan completed at {datetime.now().strftime('%Y-%m-%d %H:%M')}.\n\n"
    body += "Your Top Picks:\n\n"
    for _, row in top5.iterrows():
        body += f"• {row['Ticker']}: {row['Verdict']} (Score: {row['Score']}/100) | Price: ${row['Price']:.2f} | RS vs SPY: {row['RS vs SPY']}%\n"
        
    body += f"\nOpen the Trading Terminal to view charts and strategy levels for {top['Ticker']}."

    # Trigger Browser Notification safely
    if st.session_state.get("alert_browser"):
        st.session_state["pending_browser_notif"] = {
            "title": title, 
            "body": f"{top['Ticker']} is the top pick with score {top['Score']}/100"
        }
        
    # Email handling with active user feedback toasts
    email_enabled = st.session_state.get("alert_email", False)
    email_addr = st.session_state.get("alert_email_addr", "").strip()
    smtp_u = st.session_state.get("smtp_user", "").strip()
    smtp_p = st.session_state.get("smtp_pass", "").strip()

    if email_enabled:
        if email_addr and smtp_u and smtp_p:
            success = send_email_alert(email_addr, smtp_u, smtp_p, title, body)
            if success:
                st.toast(f"✅ Auto-scan email successfully sent to {email_addr}!", icon="✅")
            else:
                st.toast("⚠️ Auto-scan email failed. Check your App Password and SMTP details.", icon="⚠️")
        else:
            st.toast("⚠️ Auto-scan finished, but email wasn't sent because your email credentials are incomplete.", icon="⚠️")
    else:
        st.toast("✅ Auto-scan completed!", icon="✅")

@st.cache_data(ttl=60 * 60)
def fetch_ohlcv(ticker: str, period: str = "5y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        return df
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()

@st.cache_data(ttl=60 * 15)
def fetch_market_news(ticker: str = "SPY") -> List[dict]:
    try:
        t = yf.Ticker(ticker)
        # Fetching top 15 to give more of a proper feed vibe
        return t.news[:15]
    except Exception:
        return []

# ============================================================
# Technical indicators
# ============================================================
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
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
    up   = df["High"].diff()
    down = -df["Low"].diff()
    pdm  = np.where((up > down) & (up > 0), up, 0.0)
    ndm  = np.where((down > up) & (down > 0), down, 0.0)
    tr_s = atr(df, period)
    pdi  = 100 * pd.Series(pdm, index=df.index).ewm(com=period-1, adjust=False).mean() / tr_s.replace(0, np.nan)
    ndi  = 100 * pd.Series(ndm, index=df.index).ewm(com=period-1, adjust=False).mean() / tr_s.replace(0, np.nan)
    dx   = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(com=period - 1, adjust=False).mean()

def ichimoku(df: pd.DataFrame, t=9, k=26, s=52):
    ts = (df["High"].rolling(t).max() + df["Low"].rolling(t).min()) / 2
    ks = (df["High"].rolling(k).max() + df["Low"].rolling(k).min()) / 2
    sa = ((ts + ks) / 2).shift(k)
    sb = ((df["High"].rolling(s).max() + df["Low"].rolling(s).min()) / 2).shift(k)
    return ts, ks, sa, sb

def supertrend(df: pd.DataFrame, period=10, mult=3.0) -> pd.Series:
    mid   = (df["High"] + df["Low"]) / 2
    atr_s = atr(df, period)
    upper = mid + mult * atr_s
    lower = mid - mult * atr_s
    st_line = pd.Series(np.nan, index=df.index)
    trend   = pd.Series(1, index=df.index)
    for i in range(1, len(df)):
        pu, pl = upper.iloc[i-1], lower.iloc[i-1]
        upper.iloc[i] = min(upper.iloc[i], pu) if df["Close"].iloc[i-1] <= pu else upper.iloc[i]
        lower.iloc[i] = max(lower.iloc[i], pl) if df["Close"].iloc[i-1] >= pl else lower.iloc[i]
        if   df["Close"].iloc[i] > pu: trend.iloc[i] = 1
        elif df["Close"].iloc[i] < pl: trend.iloc[i] = -1
        else:                          trend.iloc[i] = trend.iloc[i-1]
        st_line.iloc[i] = lower.iloc[i] if trend.iloc[i] == 1 else upper.iloc[i]
    return st_line

def anchored_vwap(df: pd.DataFrame) -> pd.Series:
    tp  = (df["High"] + df["Low"] + df["Close"]) / 3
    cv  = df["Volume"].cumsum()
    ctv = (tp * df["Volume"]).cumsum()
    return ctv / cv.replace(0, np.nan)

def detect_hammer(df: pd.DataFrame) -> pd.Series:
    body  = (df["Close"] - df["Open"]).abs()
    lower = df[["Open","Close"]].min(axis=1) - df["Low"]
    upper = df["High"] - df[["Open","Close"]].max(axis=1)
    return (lower > 2 * body) & (upper < body)

def detect_engulfing(df: pd.DataFrame) -> pd.Series:
    return (
        (df["Close"] > df["Open"]) &
        (df["Close"].shift() < df["Open"].shift()) &
        (df["Open"] <= df["Close"].shift()) &
        (df["Close"] >= df["Open"].shift())
    )

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

def roc(series: pd.Series, period=12) -> pd.Series:
    return ((series - series.shift(period)) / series.shift(period)) * 100

def cci(df: pd.DataFrame, period=20) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    ma = tp.rolling(period).mean()
    md = (tp - ma).abs().rolling(period).mean()
    return (tp - ma) / (0.015 * md.replace(0, np.nan))

def trix(series: pd.Series, period=15) -> pd.Series:
    return ema(ema(ema(series, period), period), period).pct_change() * 100

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
    n   = len(cl)
    sar = np.zeros(n); sar[0] = lo[0]
    ep  = hi[0]; af = af_start; bull = True
    for i in range(1, n):
        if bull:
            sar[i] = sar[i-1] + af*(ep - sar[i-1])
            sar[i] = min(sar[i], lo[i-1], lo[i-2] if i >= 2 else lo[i-1])
            if lo[i] < sar[i]:
                bull=False; sar[i]=ep; ep=lo[i]; af=af_start
            elif hi[i] > ep:
                ep=hi[i]; af=min(af+af_step, af_max)
        else:
            sar[i] = sar[i-1] + af*(ep - sar[i-1])
            sar[i] = max(sar[i], hi[i-1], hi[i-2] if i >= 2 else hi[i-1])
            if hi[i] > sar[i]:
                bull=True; sar[i]=ep; ep=hi[i]; af=af_start
            elif lo[i] < ep:
                ep=lo[i]; af=min(af+af_step, af_max)
    return pd.Series(sar, index=df.index)

def fibonacci_levels(df: pd.DataFrame, lookback=120) -> Dict[str, float]:
    r = df.tail(lookback)
    hi, lo = r["High"].max(), r["Low"].min()
    d = hi - lo
    return {"0.236": hi-0.236*d, "0.382": hi-0.382*d, "0.5": hi-0.5*d, "0.618": hi-0.618*d, "0.786": hi-0.786*d}

def add_all_indicators(df: pd.DataFrame, rsi_period: int, macd_fast: int, macd_slow: int, macd_sig: int = 9) -> pd.DataFrame:
    x = df.copy()
    x["EMA20"]  = ema(x["Close"], 20)
    x["EMA50"]  = ema(x["Close"], 50)
    x["EMA200"] = ema(x["Close"], 200)
    x["RSI"]    = rsi(x["Close"], rsi_period)
    x["ATR"]    = atr(x, 14)
    x["ADX"]    = adx(x, 14)
    x["MACD"], x["MACD_SIG"], x["MACD_HIST"] = macd(x["Close"], macd_fast, macd_slow, macd_sig)
    x["OBV"]    = obv(x)
    x["CMF"]    = cmf(x)
    x["MFI"]    = mfi(x)
    x["STO_K"], x["STO_D"] = stochastic(x)
    x["WILLR"]  = williams_r(x)
    x["ROC"]    = roc(x["Close"])
    x["CCI"]    = cci(x)
    x["TRIX"]   = trix(x["Close"])
    x["AROON_UP"], x["AROON_DOWN"] = aroon(x)
    x["PSAR"]   = parabolic_sar(x)
    x["SUPER"]  = supertrend(x)
    x["BB_UP"], x["BB_MID"], x["BB_LOW"] = bollinger(x["Close"])
    x["KC_UP"], x["KC_MID"], x["KC_LOW"] = keltner(x)
    x["TENKAN"], x["KIJUN"], x["SENKOU_A"], x["SENKOU_B"] = ichimoku(x)
    x["HAMMER"] = detect_hammer(x)
    x["ENGULF"] = detect_engulfing(x)
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
    "vwap": 8, "obv": 4, "cmf": 4, "mfi": 3,
    "trend_emas": 8, "adx": 5, "ichimoku": 5, "supertrend": 5, "psar": 3,
    "rsi": 6, "macd": 6, "stoch": 3, "williams": 2, "roc": 2,
    "cci": 3, "trix": 2, "aroon": 3,
    "bb": 3, "keltner": 3, "atr_trap": 2, "donchian": 4,
    "volume_spike": 3, "gap": 1, "squeeze_release": 3,
    "hammer": 2, "engulf": 2, "fibonacci": 2,
}

REASON_MAP = {
    "vwap":           "Price is above Anchored VWAP — institutional-level support is holding.",
    "trend_emas":     "EMA 20/50/200 stack is fully bullish — multi-timeframe trend alignment confirmed.",
    "macd":           "MACD histogram is positive — upward momentum is accelerating.",
    "adx":            "ADX > 22 — market is in a sustained trend phase, not choppy sideways action.",
    "cmf":            "Chaikin Money Flow is positive — net institutional capital inflow detected.",
    "donchian":       "Price broke above the 20-day Donchian channel — fresh breakout signal.",
    "aroon":          "Aroon Up dominates Aroon Down — persistent uptrend pressure confirmed.",
    "squeeze_release":"Bollinger/Keltner squeeze is releasing to the upside — volatility expansion.",
    "ichimoku":       "Price is trading above the Ichimoku Cloud — major resistance cleared.",
    "supertrend":     "SuperTrend indicator is bullish — trend-following signal active.",
    "rsi":            "RSI is in the 50–72 sweet spot — bullish momentum without overbought risk.",
    "obv":            "On-Balance Volume is rising — volume confirms the price trend.",
    "mfi":            "Money Flow Index > 50 — net buying pressure is dominant.",
    "stoch":          "%K is crossing above %D — stochastic momentum is turning up.",
    "psar":           "Parabolic SAR is below price — trailing stop confirms the uptrend.",
    "bb":             "Price is above the Bollinger Band midline — mean-reversion bias is bullish.",
    "keltner":        "Price is above the Keltner Channel midline — trend continuation expected.",
    "volume_spike":   "Volume spike detected — unusual buyer interest confirms the move.",
    "cci":            "CCI > 0 — commodity channel momentum is in positive territory.",
    "trix":           "TRIX indicator is positive — triple-smoothed momentum supports the uptrend.",
}


def conviction_score(df: pd.DataFrame) -> Tuple[pd.Series, List[str]]:
    score = pd.Series(50.0, index=df.index)
    effects: Dict[str, pd.Series] = {}

    def apply(name: str, cond, pos: float, neg: Optional[float] = None):
        n = pos if neg is None else neg
        c = pd.Series(np.where(cond, pos, -n), index=df.index)
        effects[name] = c
        return c

    score += apply("vwap",       df["Close"] > df["AVWAP"],                   SIGNAL_WEIGHTS["vwap"])
    score += apply("obv",        df["OBV"] > df["OBV"].rolling(10).mean(),    SIGNAL_WEIGHTS["obv"])
    score += apply("cmf",        df["CMF"] > 0,                               SIGNAL_WEIGHTS["cmf"])
    score += apply("mfi",        df["MFI"] > 50,                              SIGNAL_WEIGHTS["mfi"])
    ema_bull = (df["Close"] > df["EMA20"]) & (df["EMA20"] > df["EMA50"]) & (df["EMA50"] > df["EMA200"])
    score += apply("trend_emas", ema_bull,                                    SIGNAL_WEIGHTS["trend_emas"])
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
    atr_pct = (df["ATR"] / df["Close"]).fillna(0)
    score += apply("atr_trap",   atr_pct < atr_pct.rolling(50).mean(),        SIGNAL_WEIGHTS["atr_trap"])
    score += apply("donchian",   df["Close"] > df["DONCHIAN_HIGH"].shift(1),  SIGNAL_WEIGHTS["donchian"])
    score += apply("volume_spike", df["Volume"] > 1.4*df["VOL_SMA20"],        SIGNAL_WEIGHTS["volume_spike"])
    score += apply("gap",        df["GAP_UP"],                                SIGNAL_WEIGHTS["gap"])
    score += apply(
        "squeeze_release",
        (~df["SQUEEZE_ON"]) & (df["SQUEEZE_ON"].shift(1).fillna(False)) & (df["Close"] > df["EMA20"]),
        SIGNAL_WEIGHTS["squeeze_release"], SIGNAL_WEIGHTS["squeeze_release"]/2,
    )
    for name in ["hammer", "engulf"]:
        col = "HAMMER" if name == "hammer" else "ENGULF"
        c = pd.Series(np.where(df[col], SIGNAL_WEIGHTS[name], 0), index=df.index)
        effects[name] = c; score += c
    fibs = fibonacci_levels(df)
    near = (df["Close"] - fibs["0.618"]).abs() / df["Close"] < 0.01
    c = pd.Series(np.where(near, SIGNAL_WEIGHTS["fibonacci"], 0), index=df.index)
    effects["fibonacci"] = c; score += c
    score = score.clip(0, 100)

    ranked = sorted({k: float(v.iloc[-1]) for k, v in effects.items()}.items(),
                    key=lambda kv: abs(kv[1]), reverse=True)
    reasons = []
    for name, val in ranked:
        if val <= 0:
            continue
        if name in REASON_MAP:
            reasons.append(REASON_MAP[name])
        if len(reasons) == 3:
            break
    if len(reasons) < 3:
        reasons.append("Composite multi-factor score is balanced without extreme bearish signals.")
    return score, reasons[:3]


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
    expectancy: float
    adr: float


def backtest_strategy(df: pd.DataFrame, signal: pd.Series) -> StrategyMetrics:
    pos  = np.where(signal.replace(0, np.nan).ffill().shift().fillna(0) > 0, 1, 0)
    ret  = df["Close"].pct_change().fillna(0)
    sret = pd.Series(pos, index=df.index) * ret
    eq   = (1 + sret).cumprod()
    pnl  = sret[sret != 0]
    
    gp   = pnl[pnl > 0].sum()
    gl   = -pnl[pnl < 0].sum()
    pf   = (gp / gl) if gl > 0 else np.inf
    wr   = (pnl > 0).mean() if len(pnl) else 0.0
    sh   = math.sqrt(252) * sret.mean() / sret.std() if sret.std() > 0 else 0.0
    dd   = (eq / eq.cummax() - 1).min()
    
    # Advanced KPIs
    avg_win = pnl[pnl > 0].mean() if len(pnl[pnl > 0]) else 0
    avg_loss = abs(pnl[pnl < 0].mean()) if len(pnl[pnl < 0]) else 0
    expectancy = (wr * avg_win) - ((1 - wr) * avg_loss)
    adr = ((df["High"] - df["Low"]) / df["Close"]).mean() * 100

    return StrategyMetrics(
        profit_factor  = float(pf if np.isfinite(pf) else 999.0),
        win_rate       = float(wr),
        sharpe         = float(sh),
        max_drawdown   = float(dd),
        total_return   = float(eq.iloc[-1] - 1),
        buy_hold_return= float((1 + ret).cumprod().iloc[-1] - 1),
        expectancy     = float(expectancy),
        adr            = float(adr)
    )

@st.cache_data(ttl=60 * 60, show_spinner=False)
def optimize_params(raw: pd.DataFrame, fast: bool = False) -> Tuple[Dict[str, int], StrategyMetrics, pd.DataFrame]:
    best_obj, best_params, best_metrics, best_df = None, None, None, None
    # Feature: fast_optimize for scanner so it doesn't freeze
    grid_rsi = [10, 14] if fast else [7, 9, 12, 14]
    grid_macd = [(8,21), (12,26)] if fast else [(8,21),(10,24),(12,26),(15,30)]
    
    for rp in grid_rsi:
        for f, s in grid_macd:
            if f >= s:
                continue
            tmp = add_all_indicators(raw, rp, f, s)
            sig = build_signals(tmp)
            m   = backtest_strategy(tmp, sig)
            obj = m.profit_factor * 0.6 + m.win_rate * 100 * 0.4
            if best_obj is None or obj > best_obj:
                best_obj, best_params, best_metrics, best_df = obj, {"rsi": rp, "macd_fast": f, "macd_slow": s}, m, tmp
    return best_params, best_metrics, best_df


def verdict_from_score(score: float) -> str:
    if score >= 72: return "STRONG BUY"
    if score <= 32: return "STRONG SELL"
    return "NEUTRAL"


def risk_levels(df: pd.DataFrame) -> Dict[str, float]:
    last  = df.iloc[-1]
    entry = float(last["Close"])
    atr_v = float(last["ATR"])
    return {
        "entry": entry,
        "stop":  round(entry - 2 * atr_v, 2),
        "pt1":   round(entry + 1 * atr_v, 2),
        "pt2":   round(entry + 2 * atr_v, 2),
        "pt3":   round(entry + 3 * atr_v, 2),
        "atr":   round(atr_v, 2),
        "rr1":   round(atr_v / (2 * atr_v), 2) if atr_v else 0,
    }


# ============================================================
# Chart builders
# ============================================================
def build_candlestick_chart(
    df: pd.DataFrame, ticker: str, view_period: str,
    show_ema20=True, show_ema50=True, show_ema200=True,
    show_avwap=True, show_bb=True, show_super=True,
    show_ichi=False, show_fib=False, show_psar=False,
) -> go.Figure:
    tc = _THEMES.get(st.session_state.get("theme", "TradingView"), _THEMES["TradingView"])
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name=ticker, increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ))
    if show_ema20:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA20"],  name="EMA 20",  line=dict(color="#f59e0b", width=1)))
    if show_ema50:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA50"],  name="EMA 50",  line=dict(color="#60a5fa", width=1)))
    if show_ema200:
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA200"], name="EMA 200", line=dict(color="#a78bfa", width=1, dash="dash")))
    if show_avwap:
        fig.add_trace(go.Scatter(x=df.index, y=df["AVWAP"], name="AVWAP", line=dict(color="#f87171", width=1.5, dash="dashdot")))
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_UP"],  name="BB Upper", line=dict(color="rgba(148,163,184,0.4)", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_LOW"], name="BB Lower", line=dict(color="rgba(148,163,184,0.4)", width=1, dash="dot"), fill="tonexty", fillcolor="rgba(148,163,184,0.04)"))
    if show_super:
        fig.add_trace(go.Scatter(x=df.index, y=df["SUPER"], name="SuperTrend", line=dict(color="#34d399", width=1.5)))
    if show_psar:
        fig.add_trace(go.Scatter(x=df.index, y=df["PSAR"], name="Parabolic SAR", mode="markers", marker=dict(symbol="circle", color="#fb923c", size=3)))
    if show_ichi:
        fig.add_trace(go.Scatter(x=df.index, y=df["SENKOU_A"], name="Senkou A", line=dict(color="rgba(52,211,153,0.5)", width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=df["SENKOU_B"], name="Senkou B", line=dict(color="rgba(248,113,113,0.5)", width=1), fill="tonexty", fillcolor="rgba(148,163,184,0.08)"))
        fig.add_trace(go.Scatter(x=df.index, y=df["TENKAN"], name="Tenkan", line=dict(color="#f472b6", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=df.index, y=df["KIJUN"],  name="Kijun",  line=dict(color="#fb923c", width=1, dash="dot")))
    if show_fib:
        fibs = fibonacci_levels(df)
        colors_fib = {"0.236":"#fbbf24","0.382":"#fb923c","0.5":"#f87171","0.618":"#c084fc","0.786":"#818cf8"}
        for lv, price in fibs.items():
            fig.add_hline(y=price, line_dash="dot", line_color=colors_fib.get(lv, "gold"),
                          annotation_text=f"Fib {lv}", annotation_position="right",
                          annotation_font=dict(size=10, color=colors_fib.get(lv, "gold")))

    # Feature: Infinite scroll default viewport
    end_dt = df.index[-1]
    days_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
    start_dt = end_dt - pd.Timedelta(days=days_map.get(view_period, 180))

    fig.update_layout(
        paper_bgcolor=tc["chart"], plot_bgcolor=tc["chart"],
        font=dict(color=tc["sub"], size=11),
        title=dict(text=f"<b>{ticker}</b> (Scroll back infinitely to view 5Y past history)", font=dict(color=tc["text"], size=15), pad=dict(b=8)),
        xaxis=dict(showgrid=True, gridcolor=tc["grid"], zeroline=False, rangeslider=dict(visible=False), color=tc["sub"], range=[start_dt, end_dt]),
        yaxis=dict(showgrid=True, gridcolor=tc["grid"], zeroline=False, color=tc["sub"], side="right"),
        height=520, dragmode="pan",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10),
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=60, t=60, b=0),
    )
    return fig


def build_score_chart(df, score, signal, view_period) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=score, name="Score", fill="tozeroy",
                              fillcolor="rgba(59,130,246,0.12)", line=dict(color="#3b82f6",width=2)))
    fig.add_hline(y=72, line_dash="dash", line_color="#10b981", annotation_text="STRONG BUY (72)")
    fig.add_hline(y=38, line_dash="dash", line_color="#ef4444", annotation_text="STRONG SELL (38)")
    fig.add_hline(y=50, line_dash="dot",  line_color="#4b5563")
    buys  = df.index[signal == 1]
    sells = df.index[signal == -1]
    if len(buys):
        fig.add_trace(go.Scatter(x=buys,  y=score[buys],  mode="markers", name="BUY",
                                  marker=dict(symbol="triangle-up",   color="#10b981", size=11)))
    if len(sells):
        fig.add_trace(go.Scatter(x=sells, y=score[sells], mode="markers", name="SELL",
                                  marker=dict(symbol="triangle-down",  color="#ef4444", size=11)))
    
    end_dt = df.index[-1]
    days_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
    start_dt = end_dt - pd.Timedelta(days=days_map.get(view_period, 180))

    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(family="monospace", color="#8b949e"),
        title=dict(text="Conviction Score (0–100)", font=dict(color="#e6edf3",size=13)),
        xaxis=dict(showgrid=True, gridcolor="#21262d", range=[start_dt, end_dt]),
        yaxis=dict(showgrid=True, gridcolor="#21262d", range=[0,100]),
        height=220, dragmode="pan", margin=dict(l=0,r=0,t=40,b=0),
    )
    return fig


def build_volume_chart(df, view_period) -> go.Figure:
    colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(df["Close"], df["Open"])]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], marker_color=colors, name="Vol"))
    fig.add_trace(go.Scatter(x=df.index, y=df["VOL_SMA20"], name="SMA20", line=dict(color="#f59e0b",width=1.5)))
    
    end_dt = df.index[-1]
    days_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
    start_dt = end_dt - pd.Timedelta(days=days_map.get(view_period, 180))

    fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                       font=dict(family="monospace",color="#8b949e"),
                       title=dict(text="Volume", font=dict(color="#e6edf3",size=12)),
                       xaxis=dict(showgrid=True, gridcolor="#21262d", range=[start_dt, end_dt]),
                       yaxis=dict(showgrid=True, gridcolor="#21262d"),
                       height=160, showlegend=False, dragmode="pan", margin=dict(l=0,r=0,t=35,b=0))
    return fig


def build_rsi_chart(df, view_period) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#fbbf24",width=2)))
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.08)", line_width=0)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(16,185,129,0.08)", line_width=0)
    fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", line_width=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#10b981", line_width=1)
    fig.add_hline(y=50, line_dash="dot",  line_color="#4b5563", line_width=1)
    
    end_dt = df.index[-1]
    days_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
    start_dt = end_dt - pd.Timedelta(days=days_map.get(view_period, 180))

    fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                       font=dict(family="monospace",color="#8b949e"),
                       title=dict(text="RSI  ·  <70 = neutral  ·  >70 = overbought  ·  <30 = oversold", font=dict(color="#e6edf3",size=11)),
                       xaxis=dict(showgrid=True, gridcolor="#21262d", range=[start_dt, end_dt]),
                       yaxis=dict(showgrid=True, gridcolor="#21262d", range=[0,100]),
                       height=180, dragmode="pan", margin=dict(l=0,r=0,t=40,b=0))
    return fig


def build_macd_chart(df, view_period) -> go.Figure:
    fig = go.Figure()
    colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["MACD_HIST"]]
    fig.add_trace(go.Bar(x=df.index,    y=df["MACD_HIST"], marker_color=colors, name="Histogram"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"],     name="MACD",   line=dict(color="#60a5fa",width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["MACD_SIG"], name="Signal", line=dict(color="#f87171",width=1.5)))
    
    end_dt = df.index[-1]
    days_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
    start_dt = end_dt - pd.Timedelta(days=days_map.get(view_period, 180))

    fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                       font=dict(family="monospace",color="#8b949e"),
                       title=dict(text="MACD  ·  blue line crossing above red = bullish momentum", font=dict(color="#e6edf3",size=11)),
                       xaxis=dict(showgrid=True, gridcolor="#21262d", range=[start_dt, end_dt]),
                       yaxis=dict(showgrid=True, gridcolor="#21262d"),
                       height=180, dragmode="pan", margin=dict(l=0,r=0,t=40,b=0))
    return fig


def build_stoch_chart(df, view_period) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["STO_K"], name="%K", line=dict(color="#818cf8",width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df["STO_D"], name="%D", line=dict(color="#fb923c",width=1.5)))
    fig.add_hline(y=80, line_dash="dash", line_color="#ef4444", line_width=1)
    fig.add_hline(y=20, line_dash="dash", line_color="#10b981", line_width=1)

    end_dt = df.index[-1]
    days_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
    start_dt = end_dt - pd.Timedelta(days=days_map.get(view_period, 180))

    fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                       font=dict(family="monospace",color="#8b949e"),
                       title=dict(text="Stochastic  ·  >80 = overbought  ·  <20 = oversold", font=dict(color="#e6edf3",size=11)),
                       xaxis=dict(showgrid=True, gridcolor="#21262d", range=[start_dt, end_dt]),
                       yaxis=dict(showgrid=True, gridcolor="#21262d", range=[0,100]),
                       height=180, dragmode="pan", margin=dict(l=0,r=0,t=40,b=0))
    return fig


def build_willr_chart(df, view_period) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["WILLR"], name="Williams %R", line=dict(color="#34d399",width=1.5), fill="tozeroy", fillcolor="rgba(52,211,153,0.07)"))
    fig.add_hline(y=-20, line_dash="dash", line_color="#ef4444", line_width=1, annotation_text="Overbought")
    fig.add_hline(y=-80, line_dash="dash", line_color="#10b981", line_width=1, annotation_text="Oversold")

    end_dt = df.index[-1]
    days_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
    start_dt = end_dt - pd.Timedelta(days=days_map.get(view_period, 180))

    fig.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                       font=dict(family="monospace",color="#8b949e"),
                       title=dict(text="Williams %R  ·  above -20 = overbought  ·  below -80 = oversold", font=dict(color="#e6edf3",size=11)),
                       xaxis=dict(showgrid=True, gridcolor="#21262d", range=[start_dt, end_dt]),
                       yaxis=dict(showgrid=True, gridcolor="#21262d", range=[-100,0]),
                       height=180, dragmode="pan", margin=dict(l=0,r=0,t=40,b=0))
    return fig

def build_pnl_scatter(df, signal) -> go.Figure:
    pos  = np.where(signal.replace(0, np.nan).ffill().shift().fillna(0) > 0, 1, 0)
    ret  = df["Close"].pct_change().fillna(0)
    sret = pd.Series(pos, index=df.index) * ret
    pnl = sret[sret != 0] * 100 # Convert to %
    
    colors = ["#10b981" if val > 0 else "#ef4444" for val in pnl]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pnl.index, y=pnl, mode='markers',
        marker=dict(size=8, color=colors, line=dict(width=1, color='#131722')),
        name="Trade PnL %"
    ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="#4b5563")
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font=dict(family="monospace", color="#8b949e"),
        title=dict(text="Trade Results Scatter (PnL %)", font=dict(color="#e6edf3", size=13)),
        xaxis=dict(showgrid=True, gridcolor="#21262d"),
        yaxis=dict(showgrid=True, gridcolor="#21262d"),
        height=250, margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig


# ============================================================
# Market scanner
# ============================================================
@st.cache_data(ttl=30 * 60, show_spinner=False)
def scan_universe(tickers: List[str], max_scan: int = 80, auto_tune: bool = False) -> pd.DataFrame:
    rows = []
    spy_raw = fetch_ohlcv("SPY", "1y")
    spy_ret = float((spy_raw["Close"].iloc[-1] / spy_raw["Close"].iloc[0] - 1) * 100) if spy_raw is not None and not spy_raw.empty else 0.0

    pb = st.progress(0, "Scanning universe...")
    
    # Process only requested amount
    target_tickers = tickers[:max_scan]
    
    for i, ticker in enumerate(target_tickers):
        pb.progress((i + 1) / len(target_tickers), text=f"Analyzing {ticker} ({i+1}/{len(target_tickers)})")
        try:
            # Need at least 1y for scanner so indicators don't break during auto_tune
            raw = fetch_ohlcv(ticker, "1y") 
            if raw is None or len(raw) < 120:
                continue

            if auto_tune:
                # Run fast optimal values for best guidance
                bp, _, _ = optimize_params(raw, fast=True)
                df = add_all_indicators(raw, bp["rsi"], bp["macd_fast"], bp["macd_slow"])
            else:
                df = add_all_indicators(raw, 14, 12, 26)
                
            score_s, reasons = conviction_score(df)
            last_score = float(score_s.iloc[-1])
            last_close = float(df["Close"].iloc[-1])
            last_rsi   = float(df["RSI"].iloc[-1])
            last_atr   = float(df["ATR"].iloc[-1])
            chg_1d  = float((df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1) * 100) if len(df) > 1 else 0.0
            
            # Compare vs 6mo
            idx_6mo = int(len(df) / 2) if len(df) > 126 else 0
            chg_6mo = float((df["Close"].iloc[-1] / df["Close"].iloc[idx_6mo] - 1) * 100) if len(df) > 1 else 0.0
            rs_ratio = round(chg_6mo - spy_ret, 2)
            
            rows.append({
                "Ticker":     ticker,
                "Price":      round(last_close, 2),
                "1D Chg%":    round(chg_1d, 2),
                "6M Chg%":    round(chg_6mo, 2),
                "RS vs SPY":  rs_ratio,
                "Score":      round(last_score, 1),
                "RSI":        round(last_rsi, 1),
                "ATR":        round(last_atr, 2),
                "Stop Loss":  round(last_close - 2*last_atr, 2),
                "Verdict":    verdict_from_score(last_score),
                "YF Link":    f"https://finance.yahoo.com/quote/{ticker}" # Quick shortcut to YF
            })
        except Exception:
            continue
    pb.empty()
    return pd.DataFrame(rows).sort_values(["Score","RS vs SPY"], ascending=False).reset_index(drop=True)


# ============================================================
# Main UI helpers
# ============================================================
_PCFG = {"scrollZoom": True, "displayModeBar": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

@st.cache_data(ttl=60 * 15, show_spinner=False)
def _run_analysis(t: str):
    """Fetch 5y data, silently calibrate parameters, and return everything needed to render results."""
    raw = fetch_ohlcv(t, "5y")
    if raw is None or raw.empty:
        return None, None, None, None, None, None, None

    if len(raw) >= 120:
        bp, _, df = optimize_params(raw)
    else:
        bp = {"rsi": 14, "macd_fast": 12, "macd_slow": 26}
        df = add_all_indicators(raw, 14, 12, 26)

    score, reasons = conviction_score(df)
    signal = build_signals(df)
    rl     = risk_levels(df)
    return df, score, signal, reasons, rl, bp, raw


def main():
    
    # Intercept initialization block for Local Storage loader
    if "ls_loaded" not in st.session_state:
        init_defaults() # populate defaults first
        load_res = ls_sync(action="load", ls_key="trading_term_v5", key="ls_loader")
        if load_res is None:
            st.spinner("Syncing local workspace...")
            st.stop()
        else:
            if load_res.get("status") == "loaded" and load_res.get("data"):
                # Apply restored data safely to session state
                for k, v in load_res["data"].items():
                    st.session_state[k] = v
            st.session_state["ls_loaded"] = True
            st.rerun()

    # Apply theme / font overrides
    inject_theme(st.session_state["theme"], st.session_state["font"])

    # Resolve scanner universe based on settings
    universe = get_universe(st.session_state["scan_list"])

    # Run auto-scan if due
    check_auto_scan(universe)

    # ── Sidebar Feature: Advanced Watchlists / Sector Rotation ──
    with st.sidebar:
        if st.session_state.get("layout_show_sectors", True):
            st.markdown("### 🔄 Sector Rotation Watchlist")
            sectors_df = fetch_sector_performance()
            if not sectors_df.empty:
                for _, row in sectors_df.iterrows():
                    st.metric(f"{row['Sector']} ({row['ETF']})", f"{row['1D %']:+.2f}%", f"{row['5D %']:+.2f}% (5D)")
            else:
                st.caption("Unable to fetch sector data.")

    # Auto-scan banner when it has fired
    top_t = st.session_state.get("auto_top_ticker", "")
    if top_t:
        top_s = st.session_state.get("auto_top_score", 0)
        st.info(f"Auto-scan alert: **{top_t}** is the top pick right now (Score {top_s}/100). Head to the Scanner tab for details.")

    # Added Paper Trading Tab and News Tab
    tab_analyze, tab_scan, tab_backtest, tab_paper, tab_news, tab_settings = st.tabs([
        "📈  ANALYZE",
        "🔍  SCANNER",
        "📊  BACKTEST",
        "💼  PAPER TRADING",
        "📰  NEWS",
        "⚙️  SETTINGS",
    ])

    # ════════════════════════════════════════════════════════
    # Tab 1 — Analyze
    # ════════════════════════════════════════════════════════
    with tab_analyze:
        left, right = st.columns([1, 3], gap="medium")

        with left:
            with st.expander("Workspace & Timeframe", expanded=True):
                ticker_input = st.text_input(
                    "Add Ticker to Workspace", value="",
                    placeholder="e.g. AAPL, TSLA, MSFT",
                    help="Type any US stock symbol to open a new tab. Press Enter.",
                ).strip().upper()
                
                if ticker_input and ticker_input not in st.session_state.active_tickers:
                    st.session_state.active_tickers.append(ticker_input)
                    st.rerun()

                period = st.selectbox(
                    "Default Zoom Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3,
                    help="Charts load 5 years of background data. This sets the default zoom window.",
                )
                st.session_state["az_period"] = period

            with st.expander("Chart Overlays", expanded=False):
                ov_c1, ov_c2 = st.columns(2)
                with ov_c1:
                    ov_ema20  = st.checkbox("EMA 20",         value=st.session_state["ov_ema20"],  help="20-day exponential moving average — short-term trend.")
                    ov_ema50  = st.checkbox("EMA 50",         value=st.session_state["ov_ema50"],  help="50-day EMA — medium-term trend support.")
                    ov_ema200 = st.checkbox("EMA 200",        value=st.session_state["ov_ema200"], help="200-day EMA — long-term bull/bear dividing line.")
                    ov_avwap  = st.checkbox("AVWAP",          value=st.session_state["ov_avwap"],  help="Anchored VWAP — volume-weighted average price from year start.")
                    ov_bb     = st.checkbox("Bollinger Bands",value=st.session_state["ov_bb"],     help="Upper and lower bands ±2 standard deviations around the 20-day MA.")
                with ov_c2:
                    ov_super  = st.checkbox("SuperTrend",     value=st.session_state["ov_super"],  help="ATR-based trailing stop. Green = uptrend, flips red on reversal.")
                    ov_psar   = st.checkbox("Parabolic SAR",  value=st.session_state["ov_psar"],   help="Dots flip sides when trend reverses — quick visual signal.")
                    ov_ichi   = st.checkbox("Ichimoku Cloud", value=st.session_state["ov_ichi"],   help="Japanese system: cloud shows support/resistance. Green cloud = bullish.")
                    ov_fib    = st.checkbox("Fibonacci",      value=st.session_state["ov_fib"],    help="Key retracement levels (23.6% – 78.6%) for support/resistance.")

            with st.expander("Sub-charts", expanded=False):
                sc_c1, sc_c2 = st.columns(2)
                with sc_c1:
                    sc_rsi   = st.checkbox("RSI",             value=st.session_state["sc_rsi"],   help="Relative Strength Index. Overbought >70, oversold <30.")
                    sc_macd  = st.checkbox("MACD",            value=st.session_state["sc_macd"],  help="Moving average convergence/divergence — momentum & trend.")
                    sc_vol   = st.checkbox("Volume",          value=st.session_state["sc_vol"],   help="Bar volume chart with 20-day average line.")
                with sc_c2:
                    sc_stoch = st.checkbox("Stochastic",      value=st.session_state["sc_stoch"], help="Price close position within recent range. >80 overbought, <20 oversold.")
                    sc_willr = st.checkbox("Williams %R",     value=st.session_state["sc_willr"], help="Like Stochastic but inverted. Near 0 = overbought, near -100 = oversold.")

            # Persist overlay choices
            for k, v in [("ov_ema20",ov_ema20),("ov_ema50",ov_ema50),("ov_ema200",ov_ema200),
                         ("ov_avwap",ov_avwap),("ov_bb",ov_bb),("ov_super",ov_super),
                         ("ov_psar",ov_psar),("ov_ichi",ov_ichi),("ov_fib",ov_fib),
                         ("sc_rsi",sc_rsi),("sc_macd",sc_macd),("sc_vol",sc_vol),
                         ("sc_stoch",sc_stoch),("sc_willr",sc_willr)]:
                st.session_state[k] = v

        with right:
            # TradingView-style Multiple Workspace Tabs
            if not st.session_state.active_tickers:
                st.info("Add a ticker on the left to start analyzing.")
            else:
                workspace_tabs = st.tabs(st.session_state.active_tickers)
                for idx, t in enumerate(st.session_state.active_tickers):
                    with workspace_tabs[idx]:
                        
                        # Feature: Unobtrusive Chrome-like Close Button
                        close_col1, close_col2 = st.columns([15, 1])
                        with close_col2:
                            st.markdown('<div class="close-tab-btn">', unsafe_allow_html=True)
                            if st.button("✖", key=f"close_{t}", help="Close this tab"):
                                st.session_state.active_tickers.remove(t)
                                st.rerun()
                            st.markdown('</div>', unsafe_allow_html=True)

                        with st.spinner(f"Analyzing {t} (Fetching 5y history for infinite scroll)…"):
                            df, score_series, signal, reasons, rl, bp, raw = _run_analysis(t)

                        if df is None:
                            st.error(f"No data found for **{t}**. Check the symbol and try again.")
                        else:
                            last       = df.iloc[-1]
                            last_score = float(score_series.iloc[-1])
                            verdict    = verdict_from_score(last_score)

                            cls_map  = {"STRONG BUY":"verdict-buy","STRONG SELL":"verdict-sell","NEUTRAL":"verdict-neutral"}
                            icon_map = {"STRONG BUY":"▲","STRONG SELL":"▼","NEUTRAL":"◆"}
                            
                            yf_link = f"https://finance.yahoo.com/quote/{t}"
                            st.markdown(
                                f'<div class="verdict-card {cls_map.get(verdict,"verdict-neutral")}">'
                                f'{icon_map.get(verdict,"")} &nbsp; <a href="{yf_link}" target="_blank" style="color: inherit; text-decoration: none; border-bottom: 1px dashed;" title="Open in Yahoo Finance">{t} ↗</a> &nbsp;·&nbsp; {verdict} &nbsp;·&nbsp; Score {last_score:.0f} / 100'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                            chg_pct = (last["Close"] / df["Close"].iloc[-2] - 1) * 100 if len(df) > 1 else 0
                            m1, m2, m3, m4, m5, m6 = st.columns(6)
                            m1.metric("Price", f"${last['Close']:.2f}", f"{chg_pct:+.2f}%", help="Last close price and 1-day % change.")
                            m2.metric("RSI",   f"{last['RSI']:.1f}",  help="<30 oversold · >70 overbought")
                            m3.metric("ATR",   f"${last['ATR']:.2f}", help="Average daily move in dollars — used for stop/target sizing.")
                            m4.metric("ADX",   f"{last['ADX']:.1f}",  help=">25 = strong trend · <20 = choppy range")
                            m5.metric("CMF",   f"{last['CMF']:.3f}",  help=">0 = money flowing in · <0 = money leaving")
                            m6.metric("MFI",   f"{last['MFI']:.1f}",  help="Volume-weighted RSI. <20 oversold · >80 overbought")

                            if st.session_state.get("layout_show_reasons", True):
                                st.markdown('<div style="font-size:0.68rem;letter-spacing:0.12em;color:#8b949e;margin:20px 0 6px 0">WHY THIS SIGNAL</div>', unsafe_allow_html=True)
                                for i, r in enumerate(reasons, 1):
                                    st.markdown(f'<div class="reason-box"><span style="color:#3b82f6;font-weight:700">{i}.</span> {r}</div>', unsafe_allow_html=True)

                            if st.session_state.get("layout_show_levels", True):
                                pct_stop = abs(rl["entry"] - rl["stop"]) / rl["entry"] * 100
                                pct_pt1  = (rl["pt1"] - rl["entry"]) / rl["entry"] * 100
                                pct_pt2  = (rl["pt2"] - rl["entry"]) / rl["entry"] * 100
                                pct_pt3  = (rl["pt3"] - rl["entry"]) / rl["entry"] * 100
                                rr1 = pct_pt1 / pct_stop if pct_stop else 0
                                st.markdown(f'<div style="font-size:0.68rem;letter-spacing:0.12em;color:#8b949e;margin:20px 0 4px 0">TRADE LEVELS &nbsp;<span style="color:#4b5563;font-size:0.6rem">ATR ${rl["atr"]:.2f} · R:R to T1 = {rr1:.1f}x</span></div>', unsafe_allow_html=True)
                                st.markdown(f"""
                                <div class="risk-card">
                                  <div class="risk-row"><span class="risk-label">Entry</span>
                                    <span style="font-weight:700">${rl['entry']:.2f}</span></div>
                                  <div class="risk-row"><span class="risk-label">🛑 Stop Loss</span>
                                    <span class="risk-stop">${rl['stop']:.2f} <span style="color:#6b7280;font-size:0.78rem">(-{pct_stop:.1f}%)</span></span></div>
                                  <div class="risk-row"><span class="risk-label">🎯 Target 1</span>
                                    <span class="risk-pt1">${rl['pt1']:.2f} <span style="color:#6b7280;font-size:0.78rem">(+{pct_pt1:.1f}%)</span></span></div>
                                  <div class="risk-row"><span class="risk-label">🎯 Target 2</span>
                                    <span class="risk-pt2">${rl['pt2']:.2f} <span style="color:#6b7280;font-size:0.78rem">(+{pct_pt2:.1f}%)</span></span></div>
                                  <div class="risk-row"><span class="risk-label">🎯 Target 3</span>
                                    <span class="risk-pt3">${rl['pt3']:.2f} <span style="color:#6b7280;font-size:0.78rem">(+{pct_pt3:.1f}%)</span></span></div>
                                </div>""", unsafe_allow_html=True)

                            st.markdown('<div style="margin-top:16px"></div>', unsafe_allow_html=True)

                            fig_price = build_candlestick_chart(
                                df, t, st.session_state["az_period"],
                                show_ema20=ov_ema20, show_ema50=ov_ema50, show_ema200=ov_ema200,
                                show_avwap=ov_avwap, show_bb=ov_bb, show_super=ov_super,
                                show_ichi=ov_ichi,   show_fib=ov_fib,  show_psar=ov_psar,
                            )
                            buys_px  = df.index[signal == 1]
                            sells_px = df.index[signal == -1]
                            if len(buys_px):
                                fig_price.add_trace(go.Scatter(x=buys_px, y=df["Low"][buys_px]*0.992, mode="markers", name="BUY Signal",
                                                               marker=dict(symbol="triangle-up", color="#10b981", size=11)))
                            if len(sells_px):
                                fig_price.add_trace(go.Scatter(x=sells_px, y=df["High"][sells_px]*1.008, mode="markers", name="SELL Signal",
                                                               marker=dict(symbol="triangle-down", color="#ef4444", size=11)))
                            st.plotly_chart(fig_price, use_container_width=True, config=_PCFG)

                            st.plotly_chart(build_score_chart(df, score_series, signal, st.session_state["az_period"]), use_container_width=True, config=_PCFG)

                            sub_charts = []
                            if sc_rsi:   sub_charts.append(("RSI",       build_rsi_chart(df, st.session_state["az_period"])))
                            if sc_macd:  sub_charts.append(("MACD",      build_macd_chart(df, st.session_state["az_period"])))
                            if sc_stoch: sub_charts.append(("Stoch",     build_stoch_chart(df, st.session_state["az_period"])))
                            if sc_willr: sub_charts.append(("Williams",  build_willr_chart(df, st.session_state["az_period"])))
                            for j in range(0, len(sub_charts), 2):
                                row = sub_charts[j:j+2]
                                cols = st.columns(len(row))
                                for col, (_, fig) in zip(cols, row):
                                    col.plotly_chart(fig, use_container_width=True, config=_PCFG)

                            if sc_vol:
                                st.plotly_chart(build_volume_chart(df, st.session_state["az_period"]), use_container_width=True, config=_PCFG)

    # ════════════════════════════════════════════════════════
    # Tab 2 — Scanner
    # ════════════════════════════════════════════════════════
    with tab_scan:
        sc1, sc2, sc3 = st.columns([2, 1, 1], gap="medium")
        with sc1:
            scan_all = st.checkbox("Scan Entire Universe", value=True, help="Scan every single stock in your selected list. May take longer for S&P 500.")
            max_scan = st.slider(
                "Stocks to scan (if not entire universe)", 20, 200, 80, 10,
                disabled=scan_all
            )
            auto_tune = st.checkbox("Auto-Tune Indicators per stock", value=True, help="Runs optimization per-stock to find best params perfectly guided for it.")
        with sc3:
            st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
            scan_btn = st.button("SCAN NOW", type="primary", use_container_width=True,
                                 help="Scan stocks and rank by conviction score.")

        if scan_btn:
            limit = len(universe) if scan_all else int(max_scan)
            if limit == 0:
                st.warning("Your selected universe is empty. Please select a valid list in Settings.")
            else:
                with st.spinner(f"Scanning {min(limit, len(universe))} stocks (this may take a moment)…"):
                    st.session_state["last_scan_results"] = scan_universe(tuple(universe), limit, auto_tune)

        # Displays results if a manual scan OR an auto-scan was completed
        if "last_scan_results" in st.session_state and not st.session_state["last_scan_results"].empty:
            results = st.session_state["last_scan_results"]
            
            buys  = results[results["Verdict"] == "STRONG BUY"]
            sells = results[results["Verdict"] == "STRONG SELL"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Scanned",      len(results),                                    help="Total stocks analyzed.")
            c2.metric("Strong Buys",  len(buys),                                       help="Score ≥ 72 — top bullish setups.")
            c3.metric("Strong Sells", len(sells),                                      help="Score ≤ 38 — most bearish readings.")
            c4.metric("Neutral",      len(results) - len(buys) - len(sells),   help="Mixed signals, no clear edge.")

            top5 = buys.head(5) if len(buys) >= 1 else results.head(5)
            st.markdown('<div style="font-size:0.68rem;letter-spacing:0.12em;color:#10b981;margin:22px 0 10px 0">TOP PICKS</div>', unsafe_allow_html=True)
            pick_cols = st.columns(min(len(top5), 5))
            for col, (_, row) in zip(pick_cols, top5.iterrows()):
                chg_cls   = "pick-chg-up" if row["1D Chg%"] >= 0 else "pick-chg-down"
                chg_arrow = "▲" if row["1D Chg%"] >= 0 else "▼"
                rs_color  = "#34d399" if row["RS vs SPY"] >= 0 else "#f87171"
                col.markdown(f"""
                <div class="pick-card">
                  <div class="pick-ticker">{row['Ticker']}</div>
                  <div class="pick-score">{row['Score']}</div>
                  <div style="color:#8b949e;font-size:0.65rem;margin-bottom:6px">/ 100</div>
                  <div class="pick-price">${row['Price']:.2f}</div>
                  <div class="{chg_cls}">{chg_arrow} {abs(row['1D Chg%']):.2f}%</div>
                  <div style="color:{rs_color};font-size:0.78rem;margin-top:6px">vs SPY {row['RS vs SPY']:+.1f}%</div>
                  <div style="color:#4b5563;font-size:0.68rem;margin-top:6px">RSI {row['RSI']:.0f} · Stop ${row['Stop Loss']:.2f}</div>
                </div>""", unsafe_allow_html=True)

            # Heatmap of results
            st.markdown('<div style="font-size:0.68rem;letter-spacing:0.12em;color:#8b949e;margin:28px 0 8px 0">SIGNAL HEATMAP</div>', unsafe_allow_html=True)
            results["Heatmap_Size"] = 1
            fig_hm = px.treemap(
                results, path=["Verdict", "Ticker"], values="Heatmap_Size",
                color="Score", color_continuous_scale=["#ef4444", "#4b5563", "#10b981"],
                range_color=[30, 75]
            )
            tc = _THEMES.get(st.session_state.get("theme", "TradingView"), _THEMES["TradingView"])
            fig_hm.update_layout(paper_bgcolor=tc["chart"], plot_bgcolor=tc["chart"], font=dict(color=tc["text"]), margin=dict(t=0, l=0, r=0, b=0))
            st.plotly_chart(fig_hm, use_container_width=True)

            st.markdown('<div style="font-size:0.68rem;letter-spacing:0.12em;color:#8b949e;margin:28px 0 8px 0">ALL RESULTS</div>', unsafe_allow_html=True)

            display = results.drop(columns=["Top Signal", "Heatmap_Size"], errors="ignore")
            
            st.dataframe(
                display, 
                column_config={
                    "YF Link": st.column_config.LinkColumn("Yahoo Finance", display_text="Open YF ↗"),
                    "Verdict": st.column_config.TextColumn("Verdict")
                },
                use_container_width=True, height=480
            )

    # ════════════════════════════════════════════════════════
    # Tab 3 — Backtest
    # ════════════════════════════════════════════════════════
    with tab_backtest:
        bx, by = st.columns([1, 2], gap="medium")

        with bx:
            bt_ticker = st.text_input(
                "Ticker symbol", value=st.session_state.get("bt_last_ticker", "AAPL"),
                placeholder="e.g. AAPL, TSLA, NVDA",
                key="bt_t_input",
                help="Type any US stock symbol — no need to scroll through a list.",
            ).strip().upper()

            bt_period = st.selectbox(
                "History window", ["6mo","1y","2y","5y"], index=2,
                key="bt_p",
                help="How far back to test. Longer windows give more trades and more reliable stats.",
            )

            start_cap = st.number_input(
                "Starting capital ($)", min_value=100.0, max_value=1_000_000.0,
                value=st.session_state.get("starting_capital", 5000.0),
                step=100.0, format="%.2f",
                help="How much money you would have started with. The backtest shows how it grows or shrinks.",
            )
            st.session_state["starting_capital"] = start_cap

            run_bt = st.button("RUN BACKTEST", type="primary", use_container_width=True,
                               help="Calibrates parameters specifically for this stock, then tests strategy signals on historical data.")

        with by:
            if run_bt and bt_ticker:
                st.session_state["bt_last_ticker"] = bt_ticker
                tc = _THEMES.get(st.session_state.get("theme","TradingView"), _THEMES["TradingView"])

                with st.spinner(f"Calibrating & backtesting {bt_ticker} …"):
                    raw = fetch_ohlcv(bt_ticker, bt_period)

                if raw is None or raw.empty:
                    st.error(f"No data found for **{bt_ticker}**. Check the symbol.")
                else:
                    bp, metrics, df_bt = optimize_params(raw)
                    sig_bt = build_signals(df_bt)
                    rl_bt  = risk_levels(df_bt)

                    # Dollar simulation
                    pos    = np.where(sig_bt.replace(0, np.nan).ffill().shift().fillna(0) > 0, 1, 0)
                    ret    = df_bt["Close"].pct_change().fillna(0)
                    s_eq   = (1 + pd.Series(pos, index=df_bt.index) * ret).cumprod()
                    bh_eq  = (1 + ret).cumprod()

                    strat_final  = start_cap * float(s_eq.iloc[-1])
                    bh_final     = start_cap * float(bh_eq.iloc[-1])
                    strat_pnl    = strat_final - start_cap
                    bh_pnl       = bh_final - start_cap

                    # Dollar banner
                    bang = "profit" if strat_pnl >= 0 else "loss"
                    color_bang = "#10b981" if strat_pnl >= 0 else "#ef4444"
                    st.markdown(f"""
                    <div style="background:{tc['card']};border:1px solid {tc['border']};border-radius:10px;
                                padding:18px 22px;margin-bottom:18px">
                      <div style="font-size:0.7rem;letter-spacing:0.1em;color:{tc['sub']};margin-bottom:8px">
                        ${start_cap:,.0f} STARTING CAPITAL · <a href="https://finance.yahoo.com/quote/{bt_ticker}" target="_blank" style="color:inherit;text-decoration:none;border-bottom:1px dashed;" title="Open in Yahoo Finance">{bt_ticker} ↗</a> · {bt_period.upper()}
                      </div>
                      <div style="display:flex;gap:40px;align-items:center">
                        <div>
                          <div style="font-size:0.68rem;color:{tc['sub']}">Strategy</div>
                          <div style="font-size:1.6rem;font-weight:700;color:{color_bang}">${strat_final:,.2f}</div>
                          <div style="font-size:0.85rem;color:{color_bang}">{'+' if strat_pnl>=0 else ''}{strat_pnl:+,.2f} ({metrics.total_return:.1%})</div>
                        </div>
                        <div style="color:{tc['border']};font-size:1.5rem">vs</div>
                        <div>
                          <div style="font-size:0.68rem;color:{tc['sub']}">Buy &amp; Hold</div>
                          <div style="font-size:1.6rem;font-weight:700;color:{tc['text']}">${bh_final:,.2f}</div>
                          <div style="font-size:0.85rem;color:{tc['sub']}">{'+' if bh_pnl>=0 else ''}{bh_pnl:+,.2f} ({metrics.buy_hold_return:.1%})</div>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.session_state.get("layout_show_kpis", True):
                        # Advanced KPIs
                        st.markdown("#### Advanced KPIs")
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Win Rate",        f"{metrics.win_rate:.1%}",      help="% of trades that made money.")
                        m2.metric("Profit Factor",   f"{metrics.profit_factor:.2f}", help="Total wins ÷ losses. >1.5 = strong.")
                        m3.metric("Expectancy",      f"{metrics.expectancy:.2%}",    help="Expected average return per trade.")
                        m4.metric("Sharpe",         f"{metrics.sharpe:.2f}",        help="Risk-adjusted return. >1 good · >2 excellent")
                        
                        m5, m6, m7, m8 = st.columns(4)
                        m5.metric("vs B&H",          f"{(metrics.total_return - metrics.buy_hold_return):+.1%}", help="Strategy vs Buy & Hold.")
                        m6.metric("Max Drawdown",   f"{metrics.max_drawdown:.1%}",  help="Worst peak-to-trough loss.")
                        m7.metric("Avg Daily Range", f"{metrics.adr:.2f}%", help="Current volatility.")
                        m8.metric("Market Regime", "Trending" if metrics.adr > 2.0 else "Choppy", help="Determined by ADR.")

                        # Scatter Plot for PnL
                        st.plotly_chart(build_pnl_scatter(df_bt, sig_bt), use_container_width=True, config=_PCFG)

                    if st.session_state.get("layout_show_levels", True):
                        # Trade levels
                        pct_s = abs(rl_bt["entry"]-rl_bt["stop"])/rl_bt["entry"]*100
                        pct_1 = (rl_bt["pt1"]-rl_bt["entry"])/rl_bt["entry"]*100
                        pct_2 = (rl_bt["pt2"]-rl_bt["entry"])/rl_bt["entry"]*100
                        pct_3 = (rl_bt["pt3"]-rl_bt["entry"])/rl_bt["entry"]*100
                        st.markdown(f"""
                        <div class="risk-card" style="margin:14px 0 18px 0">
                          <div style="font-size:0.65rem;color:#4b5563;margin-bottom:8px">Current levels · ATR ${rl_bt['atr']:.2f} · params auto-tuned for {bt_ticker}</div>
                          <div class="risk-row"><span class="risk-label">Entry</span><span style="font-weight:700">${rl_bt['entry']:.2f}</span></div>
                          <div class="risk-row"><span class="risk-label">🛑 Stop</span><span class="risk-stop">${rl_bt['stop']:.2f} <span style="color:#6b7280;font-size:0.78rem">(-{pct_s:.1f}%)</span></span></div>
                          <div class="risk-row"><span class="risk-label">🎯 T1</span><span class="risk-pt1">${rl_bt['pt1']:.2f} <span style="color:#6b7280;font-size:0.78rem">(+{pct_1:.1f}%)</span></span></div>
                          <div class="risk-row"><span class="risk-label">🎯 T2</span><span class="risk-pt2">${rl_bt['pt2']:.2f} <span style="color:#6b7280;font-size:0.78rem">(+{pct_2:.1f}%)</span></span></div>
                          <div class="risk-row"><span class="risk-label">🎯 T3</span><span class="risk-pt3">${rl_bt['pt3']:.2f} <span style="color:#6b7280;font-size:0.78rem">(+{pct_3:.1f}%)</span></span></div>
                        </div>""", unsafe_allow_html=True)

                    # Equity curve in dollar terms
                    s_eq_d  = s_eq  * start_cap
                    bh_eq_d = bh_eq * start_cap

                    eq_fig = go.Figure()
                    eq_fig.add_trace(go.Scatter(x=df_bt.index, y=s_eq_d,  name="Strategy ($)",
                                                 line=dict(color=tc["accent"], width=2.5)))
                    eq_fig.add_trace(go.Scatter(x=df_bt.index, y=bh_eq_d, name="Buy & Hold ($)",
                                                 line=dict(color=tc["sub"], width=1.5, dash="dash")))
                    buys_i  = df_bt.index[sig_bt == 1]
                    sells_i = df_bt.index[sig_bt == -1]
                    if len(buys_i):
                        eq_fig.add_trace(go.Scatter(x=buys_i,  y=s_eq_d[buys_i],  mode="markers",
                                                     name="Buy",  marker=dict(symbol="triangle-up",  color="#10b981", size=9)))
                    if len(sells_i):
                        eq_fig.add_trace(go.Scatter(x=sells_i, y=s_eq_d[sells_i], mode="markers",
                                                     name="Sell", marker=dict(symbol="triangle-down", color="#ef4444", size=9)))

                    eq_fig.add_hline(y=start_cap, line_dash="dot", line_color=tc["border"],
                                     annotation_text=f"${start_cap:,.0f} start", annotation_position="right",
                                     annotation_font=dict(color=tc["sub"], size=10))
                    eq_fig.update_layout(
                        paper_bgcolor=tc["chart"], plot_bgcolor=tc["chart"],
                        font=dict(color=tc["sub"], size=11),
                        title=dict(text=f"<b>{bt_ticker}</b> — Portfolio Value (${start_cap:,.0f} starting capital)",
                                   font=dict(color=tc["text"], size=13)),
                        xaxis=dict(showgrid=True, gridcolor=tc["grid"], color=tc["sub"]),
                        yaxis=dict(showgrid=True, gridcolor=tc["grid"], color=tc["sub"],
                                   tickprefix="$", side="right"),
                        height=400, dragmode="pan", margin=dict(l=0, r=70, t=55, b=0),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
                    )
                    st.plotly_chart(eq_fig, use_container_width=True, config=_PCFG)

            else:
                st.info("Enter a ticker on the left and click **RUN BACKTEST** to simulate your strategy.")

    # ════════════════════════════════════════════════════════
    # Tab 4 — Paper Trading
    # ════════════════════════════════════════════════════════
    with tab_paper:
        p1, p2 = st.columns([1, 2], gap="large")
        
        with p1:
            st.markdown("### 💼 Execute Simulation")
            st.metric("Virtual Balance", f"${st.session_state.paper_cash:,.2f}")
            
            trade_ticker = st.selectbox("Select Asset to Trade", st.session_state.active_tickers) if st.session_state.active_tickers else None
            trade_amt = st.number_input("Amount to Risk ($)", min_value=10.0, max_value=max(10.0, st.session_state.paper_cash), value=min(500.0, max(10.0, st.session_state.paper_cash)), step=100.0)
            
            tc1, tc2 = st.columns(2)
            if tc1.button("BUY", use_container_width=True, type="primary") and trade_ticker:
                raw = fetch_ohlcv(trade_ticker, "1mo")
                if not raw.empty:
                    px_cur = raw["Close"].iloc[-1]
                    shares = trade_amt / px_cur
                    if st.session_state.paper_cash >= trade_amt:
                        st.session_state.paper_cash -= trade_amt
                        if trade_ticker in st.session_state.paper_portfolio:
                            # Average price calculation
                            old_shares = st.session_state.paper_portfolio[trade_ticker]["shares"]
                            old_avg = st.session_state.paper_portfolio[trade_ticker]["avg_price"]
                            new_avg = ((old_shares * old_avg) + trade_amt) / (old_shares + shares)
                            st.session_state.paper_portfolio[trade_ticker]["shares"] += shares
                            st.session_state.paper_portfolio[trade_ticker]["avg_price"] = new_avg
                        else:
                            st.session_state.paper_portfolio[trade_ticker] = {"shares": shares, "avg_price": px_cur}
                        st.session_state.paper_history.append({"Time": datetime.now().strftime("%Y-%m-%d %H:%M"), "Action": "BUY", "Ticker": trade_ticker, "Price": px_cur, "Shares": shares, "Value": trade_amt})
                        st.success(f"Bought {trade_ticker}")
                        st.rerun()
                    else:
                        st.error("Insufficient Funds.")
                        
            if tc2.button("SELL ALL", use_container_width=True) and trade_ticker:
                if trade_ticker in st.session_state.paper_portfolio:
                    raw = fetch_ohlcv(trade_ticker, "1mo")
                    if not raw.empty:
                        px_cur = raw["Close"].iloc[-1]
                        shares = st.session_state.paper_portfolio[trade_ticker]["shares"]
                        value = shares * px_cur
                        st.session_state.paper_cash += value
                        del st.session_state.paper_portfolio[trade_ticker]
                        st.session_state.paper_history.append({"Time": datetime.now().strftime("%Y-%m-%d %H:%M"), "Action": "SELL", "Ticker": trade_ticker, "Price": px_cur, "Shares": shares, "Value": value})
                        st.success(f"Sold {trade_ticker}")
                        st.rerun()
                else:
                    st.warning(f"You don't own any {trade_ticker}.")

            st.markdown("---")
            st.markdown("### 📐 Risk Manager")
            acct_size = st.number_input("Total Account Size ($)", value=st.session_state.paper_cash, step=500.0)
            risk_pct = st.slider("Risk Limit % per Trade", 0.5, 5.0, 1.0, 0.1)
            stop_loss_pct = st.number_input("Expected Stop Loss %", value=5.0, step=0.5)
            
            risk_amt = (acct_size * risk_pct) / 100
            max_pos_size = risk_amt / (stop_loss_pct / 100) if stop_loss_pct > 0 else 0
            
            st.info(f"**Recommended Risk Profile:**\n- Max Capital at Risk: **${risk_amt:.2f}**\n- Max Position Size: **${max_pos_size:.2f}**")

        with p2:
            st.markdown("### 🗃️ Open Positions")
            if st.session_state.paper_portfolio:
                port_rows = []
                for k, v in st.session_state.paper_portfolio.items():
                    raw = fetch_ohlcv(k, "1mo")
                    current_px = raw["Close"].iloc[-1] if not raw.empty else v['avg_price']
                    pnl_pct = ((current_px / v['avg_price']) - 1) * 100
                    port_rows.append({
                        "Ticker": k,
                        "Shares": round(v['shares'], 4),
                        "Avg Entry": f"${v['avg_price']:.2f}",
                        "Current": f"${current_px:.2f}",
                        "Market Value": f"${v['shares']*current_px:.2f}",
                        "PnL %": f"{pnl_pct:+.2f}%"
                    })
                st.dataframe(pd.DataFrame(port_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No open positions. Use the terminal to buy assets.")
                
            st.markdown("### 📜 Trade History")
            if st.session_state.paper_history:
                st.dataframe(pd.DataFrame(st.session_state.paper_history), use_container_width=True, hide_index=True)
            else:
                st.caption("No trades executed yet.")


    # ════════════════════════════════════════════════════════
    # Tab 5 — News
    # ════════════════════════════════════════════════════════
    with tab_news:
        st.markdown("### 📰 Market News & Alerts")
        n_col1, n_col2 = st.columns([3, 1], gap="large")

        with n_col1:
            news_ticker = st.text_input("Search news by ticker symbol:", "SPY", help="Enter a stock ticker to fetch its latest news.").strip().upper()
            st.markdown(f"#### Latest Headlines for {news_ticker}")
            news_items = fetch_market_news(news_ticker)

            if news_items:
                for item in news_items:
                    pub_time = item.get("providerPublishTime", time.time())
                    dt_str = datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d %H:%M")
                    
                    # Use pure HTML to guarantee it opens in a new tab securely and avoids Markdown rendering bugs
                    link = item.get('link', '#')
                    title = item.get('title', 'Headline')
                    st.markdown(f'<h4><a href="{link}" target="_blank" style="color: #2962ff; text-decoration: none;">{title}</a></h4>', unsafe_allow_html=True)
                    st.caption(f"Published by **{item.get('publisher', 'Yahoo Finance')}** • {dt_str}")
                    st.divider()
            else:
                st.info("No news fetched. Check connection.")

                    img_html = f'<img src="{img_url}" style="width: 100px; height: 100px; object-fit: cover; border-radius: 6px; flex-shrink: 0;">' if img_url else f'<div style="width: 100px; height: 100px; background: #2a2e39; border-radius: 6px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; color: #787b86; font-size: 2rem;">📰</div>'

                    st.markdown(f"""
                    <a href="{item.get('link', '#')}" target="_blank" style="text-decoration:none;">
                        <div class="news-card" style="display: flex; gap: 15px; align-items: center;">
                            {img_html}
                            <div style="flex-grow: 1;">
                                <div class="news-title" style="font-size: 1.1rem; line-height: 1.3; margin-bottom: 8px;">{item.get('title', 'Headline')}</div>
                                <div class="news-meta">{item.get('publisher', 'Yahoo Finance')} • {dt_str}</div>
                            </div>
                        </div>
                    </a>
                    """, unsafe_allow_html=True)
            else:
                st.info(f"No news fetched for {news_ticker}. Check connection or try another ticker.")
                
        with n_col2:
            st.markdown("#### Global Indices Snapshot")
            with st.spinner("Fetching..."):
                try:
                    majors = yf.download("SPY QQQ DIA IWM BTC-USD", period="5d", progress=False)["Close"]
                    for c in ["SPY", "QQQ", "DIA", "IWM", "BTC-USD"]:
                        if c in majors.columns:
                            val = majors[c].iloc[-1]
                            chg = (val / majors[c].iloc[-2] - 1) * 100
                            name = "Bitcoin" if c == "BTC-USD" else c
                            st.metric(name, f"${val:,.2f}", f"{chg:+.2f}%")
                except:
                    st.caption("Unable to fetch indices.")

    # ════════════════════════════════════════════════════════
    # Tab 6 — Settings
    # ════════════════════════════════════════════════════════
    with tab_settings:
        st.markdown("### User Preferences")
        s1, s2 = st.columns(2, gap="large")

        with s1:
            st.markdown("#### Appearance & Formatting")
            all_themes = list(_THEMES.keys())
            theme_choice = st.selectbox(
                "Color theme", all_themes,
                index=all_themes.index(st.session_state.get("theme","TradingView")) if st.session_state.get("theme","TradingView") in all_themes else 0,
                help="Choose the visual style for the entire terminal.",
            )
            if theme_choice != st.session_state["theme"]:
                st.session_state["theme"] = theme_choice
                sync_settings()
                st.rerun()

            all_fonts = list(_FONT_CSS.keys())
            font_choice = st.selectbox(
                "Font", all_fonts,
                index=all_fonts.index(st.session_state.get("font","JetBrains Mono")) if st.session_state.get("font","JetBrains Mono") in all_fonts else 0,
                help="Font used throughout the interface.",
            )
            if font_choice != st.session_state["font"]:
                st.session_state["font"] = font_choice
                sync_settings()
                st.rerun()

            st.markdown("#### Layout & Workflow Customization")
            st.session_state.layout_show_reasons = st.checkbox("Show 'Why This Signal' Explanations", value=st.session_state.get("layout_show_reasons", True))
            st.session_state.layout_show_levels = st.checkbox("Show Risk/Target Levels UI", value=st.session_state.get("layout_show_levels", True))
            st.session_state.layout_show_kpis = st.checkbox("Show Advanced KPIs (Backtest Tab)", value=st.session_state.get("layout_show_kpis", True))
            st.session_state.layout_show_sectors = st.checkbox("Show Sector Watchlist (Sidebar)", value=st.session_state.get("layout_show_sectors", True))

        with s2:
            st.markdown("#### Scanner Universe")
            scan_options = ["Major ETFs & Funds", "S&P 500 + Nasdaq-100", "S&P 500", "Nasdaq-100", "Dow Jones 30", "Custom List"]
            scan_list_choice = st.selectbox(
                "Stock list", scan_options,
                index=scan_options.index(st.session_state.get("scan_list","Major ETFs & Funds")) if st.session_state.get("scan_list") in scan_options else 0,
                help="Which index to draw stocks from in the Scanner and Analyze tabs.",
            )
            if scan_list_choice != st.session_state["scan_list"]:
                st.session_state["scan_list"] = scan_list_choice
                sync_settings()
                st.rerun()

            if st.session_state["scan_list"] == "Custom List":
                custom_raw = st.text_area(
                    "Your custom tickers (comma or space separated)",
                    value=st.session_state.get("custom_tickers",""),
                    placeholder="AAPL, TSLA, NVDA, MSFT, AMZN",
                    height=100,
                    help="Enter any tickers you want to scan. They'll be used immediately in Scanner and Analyze.",
                )
                st.session_state["custom_tickers"] = custom_raw
                parsed = [t.strip().upper() for t in custom_raw.replace(","," ").split() if t.strip()]
                st.caption(f"{len(parsed)} tickers entered: {', '.join(parsed[:10])}{'…' if len(parsed)>10 else ''}")

            st.markdown("#### Alerts & Auto-Scan")
            auto_on = st.toggle(
                "Enable auto-scan",
                value=st.session_state.get("auto_scan", False),
                help="Automatically scans on a timer and alerts you to the best setup.",
            )
            st.session_state["auto_scan"] = auto_on

            if auto_on:
                interval = st.select_slider(
                    "Scan every",
                    options=[5,10,15,30,60],
                    value=st.session_state.get("scan_interval",15),
                    format_func=lambda x: f"{x} min",
                    help="How often to auto-scan. Browser tab must stay open.",
                )
                st.session_state["scan_interval"] = interval
                last_ts = st.session_state.get("last_auto_scan", 0.0)
                if last_ts > 0:
                    elapsed = int((time.time()-last_ts)/60)
                    st.caption(f"Last run {elapsed} min ago · next in ~{max(0,interval-elapsed)} min")
                else:
                    st.caption("No auto-scan has run yet this session.")

                br_on = st.checkbox("Browser pop-up notifications", value=st.session_state.get("alert_browser",False),
                    help="Browser notification when a Strong Buy signal fires.")
                st.session_state["alert_browser"] = br_on
                if br_on:
                    components.html("""
                    <div style="margin-top:4px">
                      <button onclick="Notification.requestPermission().then(p=>{
                        document.getElementById('ns').textContent=p==='granted'?'✓ Enabled':'✗ Blocked — allow in browser settings';
                        document.getElementById('ns').style.color=p==='granted'?'#34d399':'#f87171';
                      });" style="background:#2962ff;color:#fff;border:none;border-radius:6px;
                                   padding:8px 18px;cursor:pointer;font-size:0.8rem;">
                        Allow Notifications
                      </button>
                      <span id="ns" style="margin-left:10px;font-size:0.8rem;color:#787b86;"></span>
                    </div>""", height=44)

                email_on = st.checkbox("Email alerts (Gmail)", value=st.session_state.get("alert_email",False),
                    help="Email alert on Strong Buy detection. Requires a Gmail App Password.")
                st.session_state["alert_email"] = email_on
                if email_on:
                    ea1, ea2 = st.columns(2)
                    with ea1:
                        addr = st.text_input("Send to", value=st.session_state.get("alert_email_addr",""), placeholder="you@gmail.com")
                        st.session_state["alert_email_addr"] = addr
                    with ea2:
                        su = st.text_input("Gmail sender", value=st.session_state.get("smtp_user",""), placeholder="sender@gmail.com")
                        st.session_state["smtp_user"] = su
                    sp = st.text_input("App Password", value=st.session_state.get("smtp_pass",""), type="password",
                        help="16-char App Password from myaccount.google.com/apppasswords")
                    st.session_state["smtp_pass"] = sp
                    if st.button("Send test email"):
                        if addr and su and sp:
                            ok = send_email_alert(addr, su, sp, "Terminal Alert Test", "Email alerts are working correctly.")
                            st.success("Sent!") if ok else st.error("Failed — check username and App Password.")
                        else:
                            st.warning("Fill in all three email fields first.")

    # Show any pending browser notifications generated during the run
    if "pending_browser_notif" in st.session_state:
        notif = st.session_state.pop("pending_browser_notif")
        push_browser_notification(notif["title"], notif["body"])

    # ── Final Persistence Trigger ──
    # Track states down here so ANY update across the app triggers a silent local save.
    st.session_state.save_counter = st.session_state.get("save_counter", 0) + 1
    data_to_save = {k: st.session_state[k] for k in PERSISTENT_KEYS if k in st.session_state}
    ls_sync(action="save", ls_key="trading_term_v5", data=data_to_save, counter=st.session_state.save_counter, key="ls_saver")

if __name__ == "__main__":
    main()
