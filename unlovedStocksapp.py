# app.py
# Streamlit dashboard to screen for US stocks that are down over 3–6 months
# but still show healthy margins and sales growth.

import io
import math
import time
from typing import List, Tuple

import pandas as pd
import numpy as np
import streamlit as st

# External data packages
try:
    import yfinance as yf
except Exception as e:
    st.error("Failed to import yfinance. Run: pip install yfinance")
    raise

try:
    from yahooquery import Ticker
except Exception as e:
    st.error("Failed to import yahooquery. Run: pip install yahooquery")
    raise

st.set_page_config(
    page_title="Unloved but Profitable — US Equity Screener",
    page_icon="📉",
    layout="wide",
)

# --------------------------
# Helpers
# --------------------------

@st.cache_data(ttl=60 * 60)
def get_sp500_universe() -> pd.DataFrame:
    """Fetch S&P 500 constituents from Wikipedia.
    Returns a DataFrame with columns: symbol, security, sector, sub_industry
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0].rename(
        columns={
            "Symbol": "symbol",
            "Security": "security",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "sub_industry",
        }
    )
    df["symbol"] = df["symbol"].str.replace("\.", "-", regex=False)  # BRK.B -> BRK-B
    return df[["symbol", "security", "sector", "sub_industry"]]


def chunk_list(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i : i + n] for i in range(0, len(lst), n)]


@st.cache_data(ttl=60 * 60)
def get_price_returns(tickers: List[str]) -> pd.DataFrame:
    """Download 1y daily prices and compute 3m & 6m returns (Adj Close)."""
    if len(tickers) == 0:
        return pd.DataFrame()
    data = yf.download(
        tickers=tickers,
        period="1y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    rows = []
    for t in tickers:
        try:
            s = data[t]["Close"].dropna()
        except Exception:
            # Single-ticker download returns a different shape
            try:
                s = data["Close"].dropna()
            except Exception:
                continue
        if s.empty:
            continue
        last = s.iloc[-1]
        # 3m ~ 63 trading days, 6m ~ 126 trading days
        ret_3m = (last / s.iloc[-63] - 1.0) if len(s) > 63 else np.nan
        ret_6m = (last / s.iloc[-126] - 1.0) if len(s) > 126 else np.nan
        rows.append({"symbol": t, "ret_3m": ret_3m, "ret_6m": ret_6m, "last_price": last})
    return pd.DataFrame(rows)


@st.cache_data(ttl=60 * 60)
def get_fundamentals(tickers: List[str]) -> pd.DataFrame:
    """Pull quarterly income statements via yahooquery and compute TTM metrics.

    Returns columns: symbol, revenue_ttm, gross_profit_ttm, operating_income_ttm,
    net_income_ttm, gross_margin, op_margin, net_margin, rev_ttm_prev, rev_ttm_growth,
    market_cap, sector, industry
    """
    if len(tickers) == 0:
        return pd.DataFrame()

    df_list = []
    for batch in chunk_list(tickers, 30):  # keep batches modest to reduce rate limits
        tk = Ticker(batch, asynchronous=True)

        # Quarterly income statements
        try:
            inc_q = tk.income_statement(trailing=True, quarterly=True)
            if isinstance(inc_q, dict):
                # yahooquery may return error dict; coerce to empty
                inc_q = pd.DataFrame()
        except Exception:
            inc_q = pd.DataFrame()

        # Key stats / price for market cap
        try:
            key_stats = tk.key_stats
            if not isinstance(key_stats, pd.DataFrame):
                key_stats = pd.DataFrame(key_stats).T
        except Exception:
            key_stats = pd.DataFrame()

        # Profile for sector/industry
        try:
            profile = tk.summary_profile
            if not isinstance(profile, pd.DataFrame):
                profile = pd.DataFrame(profile).T
        except Exception:
            profile = pd.DataFrame()

        # Normalize fields
        mcap = (
            key_stats.reset_index()[["symbol", "marketCap"]]
            if not key_stats.empty
            else pd.DataFrame(columns=["symbol", "marketCap"])
        )
        prof = (
            profile.reset_index()[["symbol", "sector", "industry"]]
            if not profile.empty
            else pd.DataFrame(columns=["symbol", "sector", "industry"])
        )

        if isinstance(inc_q, pd.DataFrame) and not inc_q.empty:
            inc_q = inc_q.reset_index()
            # Ensure numeric
            for col in [
                "TotalRevenue",
                "GrossProfit",
                "OperatingIncome",
                "NetIncome",
            ]:
                if col in inc_q.columns:
                    inc_q[col] = pd.to_numeric(inc_q[col], errors="coerce")
            # Compute TTM by summing last 4 quarters per symbol
            ttm = (
                inc_q.sort_values(["symbol", "asOfDate"])  # asOfDate ascending
                .groupby("symbol")
                .apply(lambda g: pd.Series(
                    {
                        "revenue_ttm": g["TotalRevenue"].tail(4).sum(skipna=True),
                        "gross_profit_ttm": g["GrossProfit"].tail(4).sum(skipna=True),
                        "operating_income_ttm": g["OperatingIncome"].tail(4).sum(skipna=True),
                        "net_income_ttm": g["NetIncome"].tail(4).sum(skipna=True),
                        "rev_ttm_prev": g["TotalRevenue"].tail(8).head(4).sum(skipna=True),
                    }
                ))
                .reset_index()
            )
            # Margins and growth
            def safe_div(a, b):
                return np.nan if (b is None or b == 0 or pd.isna(b)) else a / b

            ttm["gross_margin"] = [safe_div(a, b) for a, b in zip(ttm["gross_profit_ttm"], ttm["revenue_ttm"])]
            ttm["op_margin"] = [safe_div(a, b) for a, b in zip(ttm["operating_income_ttm"], ttm["revenue_ttm"])]
            ttm["net_margin"] = [safe_div(a, b) for a, b in zip(ttm["net_income_ttm"], ttm["revenue_ttm"])]
            ttm["rev_ttm_growth"] = [
                np.nan if (pd.isna(prev) or prev == 0) else (rev - prev) / prev
                for rev, prev in zip(ttm["revenue_ttm"], ttm["rev_ttm_prev"])
            ]
        else:
            ttm = pd.DataFrame(columns=[
                "symbol",
                "revenue_ttm",
                "gross_profit_ttm",
                "operating_income_ttm",
                "net_income_ttm",
                "rev_ttm_prev",
                "gross_margin",
                "op_margin",
                "net_margin",
                "rev_ttm_growth",
            ])

        merged = (
            ttm.merge(mcap, on="symbol", how="left")
               .merge(prof, on="symbol", how="left")
        )
        df_list.append(merged)

        # Gentle sleep to be nice to endpoints
        time.sleep(0.5)

    out = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
    return out


def fmt_pct(x):
    return "" if pd.isna(x) else f"{100*x:.1f}%"


# --------------------------
# UI — Sidebar controls
# --------------------------

st.title("📉 Unloved but Profitable — US Equity Screener")

with st.sidebar:
    st.header("Universe & Filters")
    st.write("Choose your universe and set minimum fundamentals.")

    universe_choice = st.selectbox(
        "Stock universe",
        ["S&P 500 (auto)", "Upload custom tickers (CSV)"]
    )

    uploaded = None
    if universe_choice == "Upload custom tickers (CSV)":
        uploaded = st.file_uploader("Upload a CSV with a 'symbol' column", type=["csv"])    

    col_a, col_b = st.columns(2)
    with col_a:
        min_op_margin = st.number_input("Min operating margin", value=0.10, min_value=-1.0, max_value=1.0, step=0.01)
        min_rev_growth = st.number_input("Min revenue growth (TTM)", value=0.05, min_value=-1.0, max_value=2.0, step=0.01)
    with col_b:
        min_net_margin = st.number_input("Min net margin", value=0.05, min_value=-1.0, max_value=1.0, step=0.01)
        min_mcap = st.number_input("Min market cap (USD, billions)", value=5.0, min_value=0.0, step=1.0)

    unloved_window_help = "Both past 3m and 6m price returns must be negative."
    unloved_required = st.checkbox("Require 3m & 6m performance < 0", value=True, help=unloved_window_help)

    sector_filter = st.text_input("Filter sectors (comma-separated, optional)")

    run_btn = st.button("Run Screener", type="primary")

# --------------------------
# Data pipeline
# --------------------------

if run_btn:
    # Universe
    if universe_choice == "S&P 500 (auto)":
        univ = get_sp500_universe()
    else:
        if uploaded is None:
            st.warning("Please upload a CSV with a 'symbol' column.")
            st.stop()
        tmp = pd.read_csv(uploaded)
        if "symbol" not in tmp.columns:
            st.error("Your CSV must have a 'symbol' column.")
            st.stop()
        univ = tmp[["symbol"]].assign(security=np.nan, sector=np.nan, sub_industry=np.nan)

    tickers = sorted(univ["symbol"].dropna().unique().tolist())

    with st.spinner(f"Downloading prices for {len(tickers)} tickers…"):
        prices = get_price_returns(tickers)

    with st.spinner("Fetching fundamentals (TTM margins & revenue growth)…"):
        fund = get_fundamentals(tickers)

    base = (
        univ.merge(prices, on="symbol", how="left")
            .merge(fund, on=["symbol", "sector"], how="left")
    )

    # Filters
    df = base.copy()

    if unloved_required:
        df = df[(df["ret_3m"] < 0) & (df["ret_6m"] < 0)]

    df = df[(df["op_margin"] >= min_op_margin) & (df["net_margin"] >= min_net_margin)]
    df = df[(df["rev_ttm_growth"] >= min_rev_growth)]
    df = df[(df["marketCap"].fillna(0) >= min_mcap * 1e9)]

    if sector_filter.strip():
        wanted = {s.strip().lower() for s in sector_filter.split(",") if s.strip()}
        df = df[df["sector"].astype(str).str.lower().isin(wanted)]

    # Presentation
    if df.empty:
        st.warning("No matches with current filters. Try relaxing thresholds or uploading a different universe.")
        st.stop()

    # Nice columns
    view = df[[
        "symbol", "security", "sector", "last_price", "ret_3m", "ret_6m",
        "revenue_ttm", "rev_ttm_growth", "op_margin", "net_margin", "marketCap"
    ]].copy()

    view = view.rename(columns={
        "last_price": "Price",
        "ret_3m": "3m%",
        "ret_6m": "6m%",
        "revenue_ttm": "Revenue TTM",
        "rev_ttm_growth": "Rev TTM YoY%",
        "op_margin": "Op Margin%",
        "net_margin": "Net Margin%",
        "marketCap": "Mkt Cap",
    })

    # Formatting for display
    view["3m%"] = view["3m%"].apply(fmt_pct)
    view["6m%"] = view["6m%"].apply(fmt_pct)
    view["Rev TTM YoY%"] = view["Rev TTM YoY%"].apply(fmt_pct)
    view["Op Margin%"] = view["Op Margin%"].apply(fmt_pct)
    view["Net Margin%"] = view["Net Margin%"].apply(fmt_pct)
    view["Mkt Cap"] = view["Mkt Cap"].apply(lambda x: "" if pd.isna(x) else f"${x/1e9:.1f}B")
    view["Price"] = view["Price"].apply(lambda x: "" if pd.isna(x) else f"${x:,.2f}")

    st.success(f"Found {len(view)} matches.")
    st.dataframe(view, use_container_width=True)

    # Download CSV (raw numeric values)
    numeric_out = df[[
        "symbol", "security", "sector", "last_price", "ret_3m", "ret_6m",
        "revenue_ttm", "rev_ttm_growth", "op_margin", "net_margin", "marketCap"
    ]].copy()

    csv = numeric_out.to_csv(index=False).encode("utf-8")
    st.download_button("Download results (CSV)", data=csv, file_name="unloved_profitable_screen.csv", mime="text/csv")

    # Detail panel
    st.markdown("---")
    st.subheader("🔎 Drilldown")
    sel = st.selectbox("Pick a ticker to view price & key metrics", options=df["symbol"].unique())
    if sel:
        left, right = st.columns([1,1])
        with left:
            st.markdown(f"### {sel} price (1y)")
            try:
                hist = yf.download(sel, period="1y", interval="1d", auto_adjust=True, progress=False)
                st.line_chart(hist["Close"])  # Streamlit handles styling
            except Exception as e:
                st.info("Price chart unavailable.")
        with right:
            row = df[df.symbol == sel].iloc[0]
            st.metric("3m return", fmt_pct(row.get("ret_3m")))
            st.metric("6m return", fmt_pct(row.get("ret_6m")))
            st.metric("Revenue TTM YoY", fmt_pct(row.get("rev_ttm_growth")))
            st.metric("Op margin (TTM)", fmt_pct(row.get("op_margin")))
            st.metric("Net margin (TTM)", fmt_pct(row.get("net_margin")))
            mc = row.get("marketCap")
            st.metric("Market cap", "N/A" if pd.isna(mc) else f"${mc/1e9:.1f}B")

# --------------------------
# Footer notes
# --------------------------

st.caption(
    "Data: Yahoo Finance via yfinance & yahooquery. Margins are computed from TTM sums of the last 4 quarters; growth compares last 4 vs prior 4 quarters."
)


