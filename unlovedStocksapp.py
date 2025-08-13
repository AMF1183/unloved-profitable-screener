# unlovedStocksapp.py
# Streamlit dashboard to find US stocks that are "unloved" (3m & 6m price < 0)
# but still have healthy margins and sales growth (TTM).
#
# Copy into unlovedStocksapp.py, keep requirements.txt in the repo root:
# streamlit>=1.34
# yfinance>=0.2.52
# yahooquery>=2.4.1
# pandas>=2.0
# numpy>=1.25

from __future__ import annotations

import time
from typing import List

import numpy as np
import pandas as pd
import streamlit as st

# External data libs
import yfinance as yf
from yahooquery import Ticker


# --------------------------
# Streamlit page config
# --------------------------
st.set_page_config(
    page_title="Unloved but Profitable — US Equity Screener",
    page_icon="📉",
    layout="wide",
)

st.title("📉 Unloved but Profitable — US Equity Screener")
st.caption(
    "Screens US equities for negative 3–6m performance but positive TTM margins and sales growth. "
    "Data source: Yahoo Finance (prices via yfinance, fundamentals via yahooquery)."
)


# --------------------------
# Helpers
# --------------------------
def _chunk_list(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i : i + n] for i in range(0, len(lst), n)]


def _fmt_pct(x):
    return "" if pd.isna(x) else f"{100*x:.1f}%"


def _safe_div(a, b):
    return np.nan if (b is None or pd.isna(b) or b == 0) else a / b


# --------------------------
# Universe
# --------------------------
@st.cache_data(ttl=60 * 60)
def get_sp500_universe() -> pd.DataFrame:
    """Fetch S&P 500 constituents from Wikipedia."""
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
    # Normalize Yahoo tickers (BRK.B -> BRK-B, BF.B -> BF-B, etc.)
    df["symbol"] = df["symbol"].astype(str).str.replace(".", "-", regex=False)
    return df[["symbol", "security", "sector", "sub_industry"]]


# --------------------------
# Prices (3m / 6m returns)
# --------------------------
@st.cache_data(ttl=60 * 60)
def get_price_returns(tickers: List[str]) -> pd.DataFrame:
    """Download ~1y prices and compute 3m (~63d) & 6m (~126d) returns."""
    if not tickers:
        return pd.DataFrame()

    # yfinance supports batch download; auto_adjust=True gives total returns (split/div adjusted)
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
            # For multi-ticker, data[t]["Close"]; for single-ticker, data["Close"]
            s = data[t]["Close"].dropna() if isinstance(data.columns, pd.MultiIndex) else data["Close"].dropna()
        except Exception:
            continue
        if s.empty:
            continue

        last = s.iloc[-1]
        ret_3m = np.nan
        ret_6m = np.nan
        if len(s) > 63:
            ret_3m = last / s.iloc[-63] - 1.0
        if len(s) > 126:
            ret_6m = last / s.iloc[-126] - 1.0

        rows.append({"symbol": t, "ret_3m": ret_3m, "ret_6m": ret_6m, "last_price": last})

    return pd.DataFrame(rows)


# --------------------------
# Fundamentals (TTM)
# --------------------------
@st.cache_data(ttl=60 * 60)
def get_fundamentals(tickers: List[str]) -> pd.DataFrame:
    """Pull quarterly income statements via yahooquery and compute TTM metrics.
    Returns columns: symbol, revenue_ttm, gross_profit_ttm, operating_income_ttm, net_income_ttm,
    gross_margin, op_margin, net_margin, rev_ttm_growth, marketCap, sector, industry
    """
    if not tickers:
        return pd.DataFrame()

    out_frames = []

    # Keep batches modest and retry on flaky network/timeouts to reduce empty results.
    for batch in _chunk_list(tickers, 15):
        inc_q = pd.DataFrame()
        price = {}
        summary_detail = {}
        profile = {}

        # simple retry loop
        for attempt in range(3):
            try:
                tk = Ticker(batch, asynchronous=True)

                # Quarterly income statement (trailing=True returns historical quarters)
                _inc = tk.income_statement(trailing=True, quarterly=True)
                if isinstance(_inc, pd.DataFrame):
                    inc_q = _inc
                elif isinstance(_inc, dict):
                    # Some responses are dicts keyed by symbol -> dict; coerce
                    inc_q = pd.DataFrame(_inc).T.reset_index(names="symbol")
                else:
                    inc_q = pd.DataFrame()

                # For market cap, try .price first, then .summary_detail
                price = tk.price
                summary_detail = tk.summary_detail

                # Sector/industry
                profile = tk.summary_profile
                break
            except Exception:
                time.sleep(0.8)  # backoff and retry
        else:
            # All attempts failed → skip this batch
            continue

        # ------- Normalize market cap -------
        mcap_df = pd.DataFrame(columns=["symbol", "marketCap"])
        try:
            pr = price
            pr_df = pr if isinstance(pr, pd.DataFrame) else pd.DataFrame(pr).T
            pr_df = pr_df.reset_index()
            if "symbol" not in pr_df.columns and "index" in pr_df.columns:
                pr_df = pr_df.rename(columns={"index": "symbol"})
            if {"symbol", "marketCap"} <= set(pr_df.columns):
                mcap_df = pr_df[["symbol", "marketCap"]].copy()
        except Exception:
            pass

        if mcap_df.empty:
            try:
                sd = summary_detail
                sd_df = sd if isinstance(sd, pd.DataFrame) else pd.DataFrame(sd).T
                sd_df = sd_df.reset_index()
                if "symbol" not in sd_df.columns and "index" in sd_df.columns:
                    sd_df = sd_df.rename(columns={"index": "symbol"})
                if {"symbol", "marketCap"} <= set(sd_df.columns):
                    mcap_df = sd_df[["symbol", "marketCap"]].copy()
            except Exception:
                pass

        # ------- Sector / industry -------
        prof_df = pd.DataFrame(columns=["symbol", "sector", "industry"])
        try:
            pf = profile
            pf_df = pf if isinstance(pf, pd.DataFrame) else pd.DataFrame(pf).T
            pf_df = pf_df.reset_index()
            if "symbol" not in pf_df.columns and "index" in pf_df.columns:
                pf_df = pf_df.rename(columns={"index": "symbol"})
            # Ensure presence of all columns
            for col in ["sector", "industry"]:
                if col not in pf_df.columns:
                    pf_df[col] = np.nan
            if "symbol" in pf_df.columns:
                prof_df = pf_df[["symbol", "sector", "industry"]].copy()
        except Exception:
            pass

        # ------- Income statement TTM metrics -------
        if isinstance(inc_q, pd.DataFrame) and not inc_q.empty:
            inc_q = inc_q.reset_index(drop=False)
            # Normalize column names across possible variants
            rename_map = {
                "TotalRevenue": "total_revenue",
                "GrossProfit": "gross_profit",
                "OperatingIncome": "operating_income",
                "NetIncome": "net_income",
            }
            for k, v in rename_map.items():
                if k in inc_q.columns:
                    inc_q[v] = pd.to_numeric(inc_q[k], errors="coerce")
            # Yahoo sometimes uses lower-case already
            for col in ["total_revenue", "gross_profit", "operating_income", "net_income"]:
                if col not in inc_q.columns:
                    inc_q[col] = pd.to_numeric(inc_q.get(col, np.nan), errors="coerce")

            # asOfDate may be present; if not, we still group and take last rows
            if "symbol" not in inc_q.columns:
                # Some shapes use column named "ticker" or index
                if "ticker" in inc_q.columns:
                    inc_q = inc_q.rename(columns={"ticker": "symbol"})
                elif "index" in inc_q.columns:
                    inc_q = inc_q.rename(columns={"index": "symbol"})

            inc_q = inc_q.sort_values(["symbol"] + ([c for c in ["asOfDate"] if c in inc_q.columns]))

            def compute_ttm(g: pd.DataFrame) -> pd.Series:
                # take last 8 quarters; last 4 = TTM, prev 4 = previous TTM
                last4 = g.tail(4)
                prev4 = g.tail(8).head(4)
                revenue_ttm = pd.to_numeric(last4["total_revenue"], errors="coerce").sum(skipna=True)
                gross_profit_ttm = pd.to_numeric(last4["gross_profit"], errors="coerce").sum(skipna=True)
                operating_income_ttm = pd.to_numeric(last4["operating_income"], errors="coerce").sum(skipna=True)
                net_income_ttm = pd.to_numeric(last4["net_income"], errors="coerce").sum(skipna=True)
                rev_prev = pd.to_numeric(prev4["total_revenue"], errors="coerce").sum(skipna=True)
                gross_margin = _safe_div(gross_profit_ttm, revenue_ttm)
                op_margin = _safe_div(operating_income_ttm, revenue_ttm)
                net_margin = _safe_div(net_income_ttm, revenue_ttm)
                rev_growth = np.nan if (pd.isna(rev_prev) or rev_prev == 0) else (revenue_ttm - rev_prev) / rev_prev
                return pd.Series(
                    {
                        "revenue_ttm": revenue_ttm,
                        "gross_profit_ttm": gross_profit_ttm,
                        "operating_income_ttm": operating_income_ttm,
                        "net_income_ttm": net_income_ttm,
                        "rev_ttm_prev": rev_prev,
                        "gross_margin": gross_margin,
                        "op_margin": op_margin,
                        "net_margin": net_margin,
                        "rev_ttm_growth": rev_growth,
                    }
                )

            ttm = inc_q.groupby("symbol", as_index=False).apply(compute_ttm).reset_index(drop=True)
        else:
            ttm = pd.DataFrame(
                columns=[
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
                ]
            )

        # Merge everything for this batch
        merged = (
            ttm.merge(mcap_df, on="symbol", how="left")
               .merge(prof_df, on="symbol", how="left")
        )
        out_frames.append(merged)

        # Be nice to endpoints
        time.sleep(0.5)

        # ----- finalize -----
    out = pd.concat(out_frames, ignore_index=True) if out_frames else pd.DataFrame()

    # Guarantee we always return a frame with a 'symbol' column
    if out is None or out.empty:
        return pd.DataFrame({"symbol": []})

    if "symbol" not in out.columns:
        out = out.reset_index().rename(columns={"index": "symbol"})

    out["symbol"] = out["symbol"].astype(str)
    return out


# --------------------------
# Sidebar controls
# --------------------------
with st.sidebar:
    st.header("Universe & Filters")

    universe_choice = st.selectbox(
        "Stock universe",
        ["S&P 500 (auto)", "Upload custom tickers (CSV with 'symbol' column)"],
    )

    uploaded = None
    if universe_choice == "Upload custom tickers (CSV with 'symbol' column)":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

    c1, c2 = st.columns(2)
    with c1:
        min_op_margin = st.number_input("Min operating margin (TTM)", value=0.10, min_value=-1.0, max_value=1.0, step=0.01)
        min_net_margin = st.number_input("Min net margin (TTM)", value=0.05, min_value=-1.0, max_value=1.0, step=0.01)
    with c2:
        min_rev_growth = st.number_input("Min revenue growth (TTM YoY)", value=0.05, min_value=-1.0, max_value=2.0, step=0.01)
        min_mcap_b = st.number_input("Min market cap (USD, billions)", value=5.0, min_value=0.0, step=1.0)

    unloved_required = st.checkbox(
        "Require both 3m & 6m performance < 0", value=True,
        help="If checked, only shows stocks with negative returns over both 3 and 6 months."
    )

    sector_filter = st.text_input("Filter sectors (comma-separated, optional)")

    run_btn = st.button("Run Screener", type="primary")


# --------------------------
# Pipeline
# --------------------------
if run_btn:
    # Build universe
    if universe_choice == "S&P 500 (auto)":
        univ = get_sp500_universe()
    else:
        if not uploaded:
            st.warning("Please upload a CSV with a 'symbol' column.")
            st.stop()
        tmp = pd.read_csv(uploaded)
        if "symbol" not in tmp.columns:
            st.error("Your CSV must contain a 'symbol' column.")
            st.stop()
        univ = tmp[["symbol"]].copy()
        univ["security"] = np.nan
        univ["sector"] = np.nan
        univ["sub_industry"] = np.nan
        # Normalize symbols for Yahoo
        univ["symbol"] = univ["symbol"].astype(str).str.replace(".", "-", regex=False)

    tickers = sorted(univ["symbol"].dropna().astype(str).str.replace(".", "-", regex=False).unique().tolist())

    with st.spinner(f"Downloading prices for {len(tickers)} tickers…"):
        prices = get_price_returns(tickers)

    with st.spinner("Fetching fundamentals (TTM margins & revenue growth)…"):
        fund = get_fundamentals(tickers)

    # Safety: make sure 'symbol' exists in fund even if upstream changed shape
    if fund is None or fund.empty:
        fund = pd.DataFrame({"symbol": []})
    elif "symbol" not in fund.columns:
        fund = fund.reset_index().rename(columns={"index": "symbol"})
    fund["symbol"] = fund["symbol"].astype(str)


    # Safe merge only on symbol (sector strings can be inconsistent/missing)
    base = (
        univ.merge(prices, on="symbol", how="left")
            .merge(fund, on="symbol", how="left")
    )

    # Apply filters
    df = base.copy()

    if unloved_required:
        df = df[(df["ret_3m"] < 0) & (df["ret_6m"] < 0)]

    df = df[(df["op_margin"] >= min_op_margin) & (df["net_margin"] >= min_net_margin)]
    df = df[(df["rev_ttm_growth"] >= min_rev_growth)]
    df = df[(df["marketCap"].fillna(0) >= min_mcap_b * 1e9)]

    if sector_filter.strip():
        wanted = {s.strip().lower() for s in sector_filter.split(",") if s.strip()}
        df = df[df["sector"].astype(str).str.lower().isin(wanted)]

    # Present results
    if df.empty:
        st.warning("No matches with current filters. Try relaxing thresholds or confirm data fetched successfully.")
        st.stop()

    view = df[
        [
            "symbol",
            "security",
            "sector",
            "last_price",
            "ret_3m",
            "ret_6m",
            "revenue_ttm",
            "rev_ttm_growth",
            "op_margin",
            "net_margin",
            "marketCap",
        ]
    ].copy()

    view = view.rename(
        columns={
            "last_price": "Price",
            "ret_3m": "3m%",
            "ret_6m": "6m%",
            "revenue_ttm": "Revenue TTM",
            "rev_ttm_growth": "Rev TTM YoY%",
            "op_margin": "Op Margin%",
            "net_margin": "Net Margin%",
            "marketCap": "Mkt Cap",
        }
    )

    # Nicely format
    for col in ["3m%", "6m%", "Rev TTM YoY%", "Op Margin%", "Net Margin%"]:
        view[col] = view[col].apply(_fmt_pct)
    view["Mkt Cap"] = view["Mkt Cap"].apply(lambda x: "" if pd.isna(x) else f"${x/1e9:.1f}B")
    view["Price"] = view["Price"].apply(lambda x: "" if pd.isna(x) else f"${x:,.2f}")

    st.success(f"Found {len(view)} matches.")
    st.dataframe(view, use_container_width=True)

    # Download numeric (unformatted)
    numeric_out = df[
        [
            "symbol",
            "security",
            "sector",
            "last_price",
            "ret_3m",
            "ret_6m",
            "revenue_ttm",
            "rev_ttm_growth",
            "op_margin",
            "net_margin",
            "marketCap",
        ]
    ].copy()
    st.download_button(
        "Download results (CSV)",
        data=numeric_out.to_csv(index=False).encode("utf-8"),
        file_name="unloved_profitable_screen.csv",
        mime="text/csv",
    )

    # Drilldown
    st.markdown("---")
    st.subheader("🔎 Drilldown")
    sel = st.selectbox("Pick a ticker to view price & key metrics", options=df["symbol"].unique())
    if sel:
        left, right = st.columns([1, 1])
        with left:
            st.markdown(f"### {sel} price (1y)")
            try:
                hist = yf.download(sel, period="1y", interval="1d", auto_adjust=True, progress=False)
                if "Close" in hist.columns and not hist["Close"].empty:
                    st.line_chart(hist["Close"])
                else:
                    st.info("Price chart unavailable for this ticker.")
            except Exception:
                st.info("Price chart unavailable.")
        with right:
            row = df[df.symbol == sel].iloc[0]
            st.metric("3m return", _fmt_pct(row.get("ret_3m")))
            st.metric("6m return", _fmt_pct(row.get("ret_6m")))
            st.metric("Revenue TTM YoY", _fmt_pct(row.get("rev_ttm_growth")))
            st.metric("Op margin (TTM)", _fmt_pct(row.get("op_margin")))
            st.metric("Net margin (TTM)", _fmt_pct(row.get("net_margin")))
            mc = row.get("marketCap")
            st.metric("Market cap", "N/A" if pd.isna(mc) else f"${mc/1e9:.1f}B")

st.caption(
    "Notes: 3m≈63 trading days; 6m≈126 trading days. Margins computed from TTM (sum of last 4 quarters). "
    "Revenue growth compares last 4 quarters vs previous 4 quarters. Some tickers may not return complete data."
)