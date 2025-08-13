import yfinance as yf
print(yf.Ticker("AAPL").info.get("shortName"))
import time
from typing import Dict, Any, List

import pandas as pd
import yfinance as yf
import streamlit as st

st.set_page_config(page_title="Multi-Market Stock Screener", layout="wide")

DEFAULT_TICKERS = """
AAPL
MSFT
AMZN
MC.PA
SIE.DE
7203.T
6758.T
9984.T
0005.HK
0700.HK
AZN.L
HSBA.L
BP.L
""".strip()

HELP_TICKERS = """
Suffix tips:
• Paris (Euronext): .PA  e.g., MC.PA (LVMH)
• Xetra/Frankfurt:  .DE  e.g., SIE.DE (Siemens)
• Tokyo:            .T   e.g., 7203.T (Toyota)
• Hong Kong:        .HK  e.g., 0700.HK (Tencent), 0005.HK (HSBC)
• London:           .L   e.g., AZN.L, BP.L, HSBA.L
• US (NASDAQ/NYSE): plain ticker (AAPL, MSFT, AMZN)
"""

st.sidebar.header("Tickers")
user_text = st.sidebar.text_area("One per line", DEFAULT_TICKERS, height=220, help=HELP_TICKERS)
tickers: List[str] = [t.strip() for t in user_text.splitlines() if t.strip()]

col_a, col_b = st.sidebar.columns(2)
with col_a:
    period = st.selectbox("History period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
with col_b:
    interval = st.selectbox("Candle interval", ["1d", "1wk", "1mo"], index=0)

st.sidebar.caption("Tip: Missing dividend yield is computed from last 4 dividends / current price when possible.")

@st.cache_data(ttl=300)
def fetch_one(ticker: str) -> Dict[str, Any]:
    tk = yf.Ticker(ticker)
    # fast_info is quick; info can be slow or partially missing
    fast = getattr(tk, "fast_info", {}) or {}
    # yfinance >=0.2.4: get_info() is recommended (wrapped for error safety)
    try:
        info = tk.get_info() or {}
    except Exception:
        info = {}

    price = fast.get("last_price") or info.get("regularMarketPrice")
    mcap  = fast.get("market_cap")  or info.get("marketCap")
    pe    = fast.get("trailing_pe") or info.get("trailingPE")
    dy    = fast.get("dividend_yield")

    if dy is None:
        # Compute simple TTM dividend yield from last 4 cash dividends if available
        try:
            div = tk.dividends
            if isinstance(div, pd.Series) and len(div) > 0 and price:
                dy = float(div.tail(4).sum()) / float(price)
        except Exception:
            dy = None

    exch  = fast.get("exchange") or info.get("exchange")
    ccy   = fast.get("currency") or info.get("currency")
    name  = info.get("shortName") or info.get("longName") or ticker

    return {
        "Ticker": ticker,
        "Name": name,
        "Price": price,
        "Market Cap": mcap,
        "PE (TTM)": pe,
        "Dividend Yield (%)": round((dy or 0) * 100, 2) if dy else None,
        "Exchange": exch,
        "Currency": ccy,
    }

@st.cache_data(ttl=300)
def fetch_history(tickers: List[str], period: str, interval: str) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    # auto_adjust to include splits/divs, group_by="ticker" deprecated, so use columns MultiIndex
    df = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    # Ensure we return Close as a tidy DataFrame
    if isinstance(df, pd.DataFrame) and "Close" in df.columns:
        close = df["Close"].copy()
        if isinstance(close.columns, pd.MultiIndex):
            # After recent yfinance versions, columns might be a flat Index if single ticker
            close.columns = [c[1] if isinstance(c, tuple) else c for c in close.columns.to_list()]
        return close
    return pd.DataFrame()

# --------- UI ----------
st.title("📈 Multi-Market Stock Screener (Streamlit)")
st.caption("Pulls live snapshot data from Yahoo Finance via yfinance. Cached for 5 minutes.")

run = st.button("Fetch data", type="primary", use_container_width=True)

if run:
    with st.spinner("Fetching fundamentals…"):
        rows: List[Dict[str, Any]] = []
        for t in tickers:
            try:
                rows.append(fetch_one(t))
            except Exception as e:
                rows.append({"Ticker": t, "Error": str(e)})
            # small sleep to be polite to upstream
            time.sleep(0.05)
        df = pd.DataFrame(rows)

    # Display table
    st.subheader("Snapshot")
    if df.empty:
        st.info("No data returned. Check tickers.")
    else:
        # Numeric formatting
        def fmt_int(x):
            try:
                return f"{int(x):,}"
            except Exception:
                return x

        def fmt_float(x, digits=2):
            try:
                return f"{float(x):,.{digits}f}"
            except Exception:
                return x

        show = df.copy()
        if "Price" in show:
            show["Price"] = show["Price"].map(lambda v: fmt_float(v, 2))
        if "Market Cap" in show:
            show["Market Cap"] = show["Market Cap"].map(fmt_int)
        if "PE (TTM)" in show:
            show["PE (TTM)"] = show["PE (TTM)"].map(lambda v: fmt_float(v, 2))
        if "Dividend Yield (%)" in show:
            show["Dividend Yield (%)"] = show["Dividend Yield (%)"].map(lambda v: fmt_float(v, 2))

        st.dataframe(show, use_container_width=True)

        # Quick charts
        st.subheader("Charts")
        c1, c2 = st.columns(2, gap="large")

        with c1:
            mc = df[["Ticker", "Market Cap"]].dropna().sort_values("Market Cap", ascending=False).set_index("Ticker")
            if not mc.empty:
                st.caption("Market cap (largest on top)")
                st.bar_chart(mc)

        with c2:
            scatter = df.dropna(subset=["PE (TTM)", "Market Cap"])
            if not scatter.empty:
                st.caption("PE vs Market Cap")
                st.scatter_chart(scatter, x="PE (TTM)", y="Market Cap", size=None, color="Ticker")

    # Price history (optional)
    st.subheader("Price history (Close)")
    hist = fetch_history(tickers, period, interval)
    if hist.empty:
        st.info("No history for selected symbols/period.")
    else:
        st.line_chart(hist)

else:
    st.info("Enter tickers in the sidebar, then click **Fetch data**.")
    st.code(HELP_TICKERS.strip(), language="text")
