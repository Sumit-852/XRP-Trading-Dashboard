"""
XRP/EUR Trading Dashboard — Reads state from CI/CD pipeline.
No model logic here — just visualization.

Usage: streamlit run xrp_dashboard.py
"""
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

DASH_STATE_FILE = "xrp_dashboard_state.json"
SIGNAL_FILE = "xrp_live_signal.json"
NEWS_FILE = "xrp_news_sentiment.json"
INITIAL_CAPITAL = 5_000.0

st.set_page_config(page_title="XRP/EUR AI Trading Dashboard", layout="wide")


def load_state():
    if os.path.exists(DASH_STATE_FILE):
        with open(DASH_STATE_FILE) as f:
            return json.load(f)
    return {"capital": INITIAL_CAPITAL, "xrp_held": 0, "position": "CASH",
            "trades": [], "equity_history": [], "entry_price": 0, "peak_price": 0}


def load_signal():
    if os.path.exists(SIGNAL_FILE):
        with open(SIGNAL_FILE) as f:
            return json.load(f)
    return {}


def main():
    st.title("XRP/EUR AI Trading Dashboard")
    st.caption("Model runs every 5 minutes via GitHub Actions CI/CD")

    state = load_state()
    sig_data = load_signal()
    signal = sig_data.get("signal", {})
    news = signal.get("news", {})
    vpa = signal.get("vpa", {})
    season = signal.get("seasonality", {})
    tech = signal.get("technicals", {})
    geo_dip = news.get("geo_dip", {})

    price = signal.get("price", 0)
    trades = state.get("trades", [])
    sells = [t for t in trades if t["action"] == "SELL"]
    wins = [t for t in sells if t.get("pnl_pct", 0) > 0]
    losses = [t for t in sells if t.get("pnl_pct", 0) <= 0]

    portfolio_value = state.get("capital", 0) + state.get("xrp_held", 0) * price
    pnl = (portfolio_value / INITIAL_CAPITAL - 1) * 100 if INITIAL_CAPITAL > 0 else 0
    win_rate = len(wins) / len(sells) * 100 if sells else 0

    # ── Top metrics ──────────────────────────────────────────────────
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
    col1.metric("Portfolio", f"€{portfolio_value:,.0f}", f"{pnl:+.2f}%")
    col2.metric("XRP Price", f"€{price:.4f}")
    col3.metric("Position", state.get("position", "?"),
                f"€{state.get('entry_price', 0):.4f}" if state.get("position") == "LONG" else None)
    col4.metric("Signal", signal.get("action", "?"), f"Score: {signal.get('hybrid_score', '?')}")
    col5.metric("\U0001f7e2 Wins", len(wins),
                f"+€{sum(t['eur'] * t['pnl_pct']/100 for t in wins):,.0f}" if wins else None)
    col6.metric("\U0001f534 Losses", len(losses),
                f"-€{abs(sum(t['eur'] * t['pnl_pct']/100 for t in losses)):,.0f}" if losses else None)
    col7.metric("Win Rate", f"{win_rate:.0f}%" if sells else "—",
                f"{len(wins)}/{len(sells)}" if sells else None)
    col8.metric("News", news.get("sentiment", "?"), f"{news.get('score', 0):+.2f}")

    # ── Signal panels ────────────────────────────────────────────────
    st.subheader("Model Intelligence")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("**XGBoost**")
        prob = signal.get("xgb_prob", 0) * 100
        st.metric("Prediction", signal.get("xgb_pred", "?"), f"{prob:.1f}% UP")
        st.metric("Hybrid Score", signal.get("hybrid_score", "?"))
        st.metric("Data Bars", f"{sig_data.get('data_bars', 0):,}")
        ts = sig_data.get("timestamp", "")
        st.metric("Last Update", ts[:19].replace("T", " ") if ts else "?")

    with c2:
        st.markdown("**Volume (VPA)**")
        vpa_icon = "\U0001f7e2" if vpa.get("phase") == "ACCUMULATION" else ("\U0001f534" if vpa.get("phase") == "DISTRIBUTION" else "⚪")
        st.metric("Phase", f"{vpa_icon} {vpa.get('phase', '?')}")
        st.metric("Up/Down Ratio", f"{vpa.get('up_down_ratio', 0):.2f}")
        st.metric("Health", f"{vpa.get('health', 0):+.4f}")
        st.metric("Vol Ratio", f"{vpa.get('vol_ratio', 0):.2f}x")

    with c3:
        st.markdown("**Seasonality**")
        st.metric("Month Score", f"{season.get('month_score', 0):+.1f}")
        st.metric("Quarter Score", f"{season.get('quarter_score', 0):+.1f}")
        st.metric("Halving Phase", f"{season.get('halving_phase', 0):.3f}")
        zone = season.get("halving_zone", "?")
        zone_icon = "\U0001f7e2" if zone == "PEAK" else ("\U0001f534" if zone == "LATE" else "\U0001f7e1")
        st.metric("Cycle Zone", f"{zone_icon} {zone}")

    with c4:
        st.markdown("**News Sentiment**")
        ns = news.get("score", 0)
        ns_icon = "\U0001f7e2" if ns > 0.15 else ("\U0001f534" if ns < -0.15 else "⚪")
        st.metric("Sentiment", f"{ns_icon} {news.get('sentiment', '?')}", f"{ns:+.2f}")
        st.metric("Bull / Bear", f"{news.get('bullish_pct', 0):.0f}% / {news.get('bearish_pct', 0):.0f}%")

        dip_phase = geo_dip.get("phase", "NONE")
        dip_icon = "\U0001f4b0" if dip_phase == "BUY THE DIP" else ("\U0001f504" if dip_phase == "RECOVERING" else ("\U0001f6a8" if dip_phase == "PANIC SELLING" else "—"))
        st.metric("Geo Dip", f"{dip_icon} {dip_phase}")
        st.metric("Stealth Accum", "YES" if news.get("stealth_accumulation") else "No")

    # ── Feed breakdown ───────────────────────────────────────────────
    with st.expander("News Feed Breakdown", expanded=False):
        feed_scores = news.get("feed_scores", {})
        feed_labels = {"xrp": "XRP/Ripple", "crypto_macro": "Crypto Market",
                       "geopolitical": "Geopolitical", "fed_rates": "Fed/Rates"}
        for feed, score in feed_scores.items():
            label = "BULL" if score > 0.1 else ("BEAR" if score < -0.1 else "NEUT")
            icon = "\U0001f7e2" if score > 0.1 else ("\U0001f534" if score < -0.1 else "⚪")
            st.markdown(f"{icon} **{feed_labels.get(feed, feed)}**: {score:+.3f} ({label})")

        ncol1, ncol2 = st.columns(2)
        with ncol1:
            st.markdown("**Top Bullish**")
            for h in news.get("top_bullish", []):
                title = h if isinstance(h, str) else h.get("title", "")
                st.markdown(f"- {title[:100]}")
        with ncol2:
            st.markdown("**Top Bearish**")
            for h in news.get("top_bearish", []):
                title = h if isinstance(h, str) else h.get("title", "")
                st.markdown(f"- {title[:100]}")

    # ── Equity curve ─────────────────────────────────────────────────
    st.subheader("Portfolio Equity Curve")
    eq_history = state.get("equity_history", [])
    if len(eq_history) > 1:
        eq_df = pd.DataFrame(eq_history)
        eq_df["datetime"] = pd.to_datetime(eq_df["time"], format="mixed", utc=True)

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=eq_df["datetime"], y=eq_df["value"], name="Portfolio",
            line=dict(color="#26a69a", width=2), fill="tozeroy",
            fillcolor="rgba(38,166,154,0.1)",
        ))
        fig_eq.add_hline(y=INITIAL_CAPITAL, line_dash="dash", line_color="white", opacity=0.3,
                         annotation_text=f"Starting Capital €{INITIAL_CAPITAL:,.0f}")
        fig_eq.update_layout(
            height=350, template="plotly_dark",
            yaxis_title="EUR", margin=dict(l=50, r=50, t=30, b=30),
        )
        st.plotly_chart(fig_eq, use_container_width=True)
    else:
        st.info("Equity curve will appear after a few trading cycles.")

    # ── Trade history ────────────────────────────────────────────────
    st.subheader("Trade History")
    if sells:
        trade_df = pd.DataFrame(sells)
        trade_df["time"] = pd.to_datetime(trade_df["time"]).dt.strftime("%Y-%m-%d %H:%M")
        cols_show = ["time", "price", "xrp", "eur", "pnl_pct", "reason"]
        cols_present = [c for c in cols_show if c in trade_df.columns]
        st.dataframe(
            trade_df[cols_present].sort_index(ascending=False),
            use_container_width=True, hide_index=True,
        )

        scol1, scol2, scol3, scol4 = st.columns(4)
        scol1.metric("Win Rate", f"{len(wins)}/{len(sells)} ({win_rate:.1f}%)")
        scol2.metric("Avg Win", f"+{np.mean([t['pnl_pct'] for t in wins]):.2f}%" if wins else "N/A")
        scol3.metric("Avg Loss", f"{np.mean([t['pnl_pct'] for t in losses]):.2f}%" if losses else "N/A")
        scol4.metric("Total P&L", f"€{portfolio_value - INITIAL_CAPITAL:+,.2f}")
    else:
        st.info("No trades yet. The model will trade when conditions align.")

    # ── Technical indicators ─────────────────────────────────────────
    with st.expander("Technical Indicators", expanded=False):
        tech_data = {
            "Indicator": ["EMA50", "EMA200", "MACD", "MACD Signal", "RSI14", "ATR14", "BB Position"],
            "Value": [
                f"€{tech.get('ema50', 0):.5f}",
                f"€{tech.get('ema200', 0):.5f}",
                f"{tech.get('macd', 0):.6f}",
                f"{tech.get('macd_signal', 0):.6f}",
                f"{tech.get('rsi14', 0):.1f}",
                f"{tech.get('atr14', 0):.5f}",
                f"{tech.get('bb_position', 0):.3f}",
            ],
        }
        st.table(pd.DataFrame(tech_data))

    # ── Model Training & Trade Performance ─────────────────────────
    st.markdown("---")
    st.subheader("Model Training & Trade Performance")

    data_bars = sig_data.get("data_bars", 0)
    retrained = sig_data.get("retrained", False)
    total_trades = len(trades)
    buys = [t for t in trades if t["action"] == "BUY"]

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Training Bars", f"{data_bars:,}",
               f"{data_bars * 5 / 60:.0f} hours" if data_bars else None)
    mc2.metric("Last Retrained", "This cycle" if retrained else "Previous cycle")
    mc3.metric("Total Trades", total_trades, f"{len(buys)} BUY / {len(sells)} SELL")
    mc4.metric("Total Wins", len(wins),
               f"Avg: +{np.mean([t['pnl_pct'] for t in wins]):.2f}%" if wins else None)
    mc5.metric("Total Losses", len(losses),
               f"Avg: {np.mean([t['pnl_pct'] for t in losses]):.2f}%" if losses else None)

    if sells:
        with st.expander("Win & Loss Breakdown", expanded=True):
            wcol, lcol = st.columns(2)
            with wcol:
                st.markdown("**Winning Trades**")
                if wins:
                    win_df = pd.DataFrame(wins)
                    win_df["time"] = pd.to_datetime(win_df["time"]).dt.strftime("%m-%d %H:%M")
                    win_df["pnl_pct"] = win_df["pnl_pct"].apply(lambda x: f"+{x:.2f}%")
                    st.dataframe(
                        win_df[["time", "price", "pnl_pct", "reason"]],
                        use_container_width=True, hide_index=True,
                    )
                    total_win_eur = sum(t["eur"] * t["pnl_pct"] / 100 for t in wins)
                    st.success(f"Total profit from wins: +€{total_win_eur:,.2f}")
                else:
                    st.info("No winning trades yet.")

            with lcol:
                st.markdown("**Losing Trades**")
                if losses:
                    loss_df = pd.DataFrame(losses)
                    loss_df["time"] = pd.to_datetime(loss_df["time"]).dt.strftime("%m-%d %H:%M")
                    loss_df["pnl_pct"] = loss_df["pnl_pct"].apply(lambda x: f"{x:.2f}%")
                    st.dataframe(
                        loss_df[["time", "price", "pnl_pct", "reason"]],
                        use_container_width=True, hide_index=True,
                    )
                    total_loss_eur = sum(t["eur"] * t["pnl_pct"] / 100 for t in losses)
                    st.error(f"Total loss: -€{abs(total_loss_eur):,.2f}")
                else:
                    st.info("No losing trades yet.")

        best = max(sells, key=lambda t: t.get("pnl_pct", 0))
        worst = min(sells, key=lambda t: t.get("pnl_pct", 0))
        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric("Best Trade", f"+{best['pnl_pct']:.2f}%", best.get("reason", "")[:30])
        bc2.metric("Worst Trade", f"{worst['pnl_pct']:.2f}%", worst.get("reason", "")[:30])
        profit_factor = abs(sum(t["pnl_pct"] for t in wins) / sum(t["pnl_pct"] for t in losses)) if losses and any(t["pnl_pct"] < 0 for t in losses) else 0
        bc3.metric("Profit Factor", f"{profit_factor:.2f}" if profit_factor else "—")
        avg_bars = np.mean([t.get("score", 0) for t in sells]) if sells else 0
        bc4.metric("Avg Exit Score", f"{avg_bars:.1f}")

    # ── Auto-refresh ─────────────────────────────────────────────────
    st.markdown("---")
    refresh = st.selectbox("Auto-refresh interval", ["Off", "1 min", "5 min", "15 min"], index=2)
    refresh_map = {"Off": None, "1 min": 60, "5 min": 300, "15 min": 900}
    refresh_sec = refresh_map[refresh]

    st.caption(f"Last update: {sig_data.get('timestamp', '?')[:19]} UTC | "
               f"CI/CD: every 5 min via GitHub Actions")

    if refresh_sec:
        time.sleep(refresh_sec)
        st.rerun()


if __name__ == "__main__":
    main()
