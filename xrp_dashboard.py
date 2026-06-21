"""
XRP/EUR Trading Dashboard — Reads state from CI/CD pipeline.
No model logic here — just visualization.

Usage: streamlit run xrp_dashboard.py
"""
import json
import os
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

CET = ZoneInfo("Europe/Berlin")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

DASH_STATE_FILE = "xrp_dashboard_state.json"
SIGNAL_FILE = "xrp_live_signal.json"
NEWS_FILE = "xrp_news_sentiment.json"
FEEDBACK_FILE = "xrp_trade_feedback.json"
OHLCV_FILE = "xrp_eur_5min.csv"
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
        ts_cet = pd.to_datetime(ts, utc=True).tz_convert(CET).strftime("%H:%M CET") if ts else "?"
        st.metric("Last Update", ts_cet)

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
        eq_df["datetime"] = pd.to_datetime(eq_df["time"], format="mixed", utc=True).dt.tz_convert(CET)

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

    # ── Price + Volume + Model Predictions ─────────────────────────
    st.subheader("Price & Volume with Model Predictions")
    ohlcv_bars = st.selectbox("Show last", ["6 hours", "12 hours", "24 hours", "3 days", "7 days"], index=3)
    bars_map = {"6 hours": 72, "12 hours": 144, "24 hours": 288, "3 days": 864, "7 days": 2016}
    show_bars = bars_map[ohlcv_bars]

    if os.path.exists(OHLCV_FILE):
        ohlcv = pd.read_csv(OHLCV_FILE).tail(show_bars)
        ohlcv["datetime"] = pd.to_datetime(ohlcv["time"], unit="s", utc=True).dt.tz_convert(CET)

        eq_with_pred = [e for e in eq_history if "xgb_prob" in e]
        has_predictions = len(eq_with_pred) > 2

        rows = 3 if has_predictions else 2
        heights = [0.5, 0.2, 0.3] if has_predictions else [0.65, 0.35]
        subtitles = ["XRP/EUR Price (5-min candles)", "Volume"]
        if has_predictions:
            subtitles.append("Model Prediction (UP probability)")

        fig_pv = make_subplots(
            rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.04,
            row_heights=heights, subplot_titles=subtitles,
        )

        fig_pv.add_trace(go.Candlestick(
            x=ohlcv["datetime"], open=ohlcv["open"], high=ohlcv["high"],
            low=ohlcv["low"], close=ohlcv["close"], name="OHLC",
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
        ), row=1, col=1)

        ema20 = ohlcv["close"].ewm(span=20, adjust=False).mean()
        ema50 = ohlcv["close"].ewm(span=50, adjust=False).mean()
        fig_pv.add_trace(go.Scatter(
            x=ohlcv["datetime"], y=ema20, name="EMA 20",
            line=dict(color="#ffa726", width=1), opacity=0.7,
        ), row=1, col=1)
        fig_pv.add_trace(go.Scatter(
            x=ohlcv["datetime"], y=ema50, name="EMA 50",
            line=dict(color="#ab47bc", width=1), opacity=0.7,
        ), row=1, col=1)

        buy_trades = [t for t in trades if t["action"] == "BUY"]
        sell_trades = [t for t in trades if t["action"] == "SELL"]
        if buy_trades:
            buy_df = pd.DataFrame(buy_trades)
            buy_df["dt"] = pd.to_datetime(buy_df["time"], format="mixed", utc=True).dt.tz_convert(CET)
            buy_df = buy_df[buy_df["dt"] >= ohlcv["datetime"].min()]
            if not buy_df.empty:
                fig_pv.add_trace(go.Scatter(
                    x=buy_df["dt"], y=buy_df["price"], mode="markers", name="BUY",
                    marker=dict(symbol="triangle-up", size=14, color="#26a69a", line=dict(width=1, color="white")),
                ), row=1, col=1)
        if sell_trades:
            sell_df = pd.DataFrame(sell_trades)
            sell_df["dt"] = pd.to_datetime(sell_df["time"], format="mixed", utc=True).dt.tz_convert(CET)
            sell_df = sell_df[sell_df["dt"] >= ohlcv["datetime"].min()]
            if not sell_df.empty:
                fig_pv.add_trace(go.Scatter(
                    x=sell_df["dt"], y=sell_df["price"], mode="markers", name="SELL",
                    marker=dict(symbol="triangle-down", size=14, color="#ef5350", line=dict(width=1, color="white")),
                ), row=1, col=1)

        colors = ["#26a69a" if c >= o else "#ef5350" for o, c in zip(ohlcv["open"], ohlcv["close"])]
        fig_pv.add_trace(go.Bar(
            x=ohlcv["datetime"], y=ohlcv["volume"], name="Volume",
            marker_color=colors, opacity=0.7, showlegend=False,
        ), row=2, col=1)

        vol_avg = ohlcv["volume"].rolling(50).mean()
        fig_pv.add_trace(go.Scatter(
            x=ohlcv["datetime"], y=vol_avg, name="Vol Avg",
            line=dict(color="#ffa726", width=1, dash="dot"), opacity=0.6,
        ), row=2, col=1)

        if has_predictions:
            pred_df = pd.DataFrame(eq_with_pred)
            pred_df["datetime"] = pd.to_datetime(pred_df["time"], format="mixed", utc=True).dt.tz_convert(CET)
            pred_df = pred_df[pred_df["datetime"] >= ohlcv["datetime"].min()]
            if not pred_df.empty:
                prob_colors = ["#26a69a" if p > 0.5 else "#ef5350" for p in pred_df["xgb_prob"]]
                fig_pv.add_trace(go.Bar(
                    x=pred_df["datetime"], y=pred_df["xgb_prob"] - 0.5,
                    name="UP Prob", marker_color=prob_colors, opacity=0.6,
                    base=0.5,
                ), row=3, col=1)
                fig_pv.add_hline(y=0.5, line_dash="dash", line_color="white", opacity=0.4, row=3, col=1)
                score_norm = pred_df["hybrid_score"].clip(-2, 8) / 8
                fig_pv.add_trace(go.Scatter(
                    x=pred_df["datetime"], y=score_norm, name="Hybrid Score",
                    line=dict(color="#ffa726", width=2),
                ), row=3, col=1)

        chart_height = 800 if has_predictions else 550
        fig_pv.update_layout(
            height=chart_height, template="plotly_dark",
            margin=dict(l=50, r=50, t=40, b=30),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        fig_pv.update_yaxes(title_text="EUR", row=1, col=1)
        fig_pv.update_yaxes(title_text="Vol", row=2, col=1)
        if has_predictions:
            fig_pv.update_yaxes(title_text="Prob", row=3, col=1, range=[0, 1])
        st.plotly_chart(fig_pv, use_container_width=True)

        if not has_predictions:
            st.caption("Model prediction overlay will appear after a few more cycles (accumulating xgb_prob data).")
    else:
        st.info("OHLCV data not available yet.")

    # ── Trade Feedback / Model Learning ──────────────────────────────
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE) as f:
            fb = json.load(f)

        st.subheader("Model Learning (Trade Feedback)")
        fb1, fb2, fb3, fb4, fb5, fb6 = st.columns(6)
        fb1.metric("Win Rate", f"{fb.get('overall_win_rate', 0):.0%}",
                    f"Recent: {fb.get('recent_win_rate', 0):.0%}")
        fb2.metric("Loss Streak", fb.get("loss_streak", 0),
                    "Caution" if fb.get("loss_streak", 0) >= 3 else "OK")
        fb3.metric("Score Threshold", fb.get("min_score_effective", 3),
                    f"+{fb.get('score_adjustment', 0)} from default" if fb.get("score_adjustment", 0) else "Default")
        fb4.metric("Adapted SL", f"{fb.get('adapted_sl_pct', 1.0)}%",
                    f"Default: {1.0}%" if fb.get("adapted_sl_pct", 1.0) != 1.0 else None)
        fb5.metric("Adapted TP", f"{fb.get('adapted_tp_pct', 4.0)}%",
                    f"Default: {4.0}%" if fb.get("adapted_tp_pct", 4.0) != 4.0 else None)
        fb6.metric("Avg Win / Loss",
                    f"+{fb.get('avg_win_pct', 0):.2f}%",
                    f"/ {fb.get('avg_loss_pct', 0):.2f}%")

        with st.expander("Loss Analysis", expanded=False):
            la1, la2 = st.columns(2)
            with la1:
                st.markdown("**Loss Reasons**")
                for reason, count in fb.get("loss_reasons", {}).items():
                    st.markdown(f"- {reason}: **{count}** trades")
            with la2:
                st.markdown("**Risk Profile**")
                st.markdown(f"- Stop Loss exits: **{fb.get('sl_loss_ratio', 0):.0%}** of losses")
                st.markdown(f"- Model Sell exits: **{fb.get('model_loss_ratio', 0):.0%}** of losses")
                st.markdown(f"- Low-score entries losing: **{fb.get('low_score_loss_pct', 0):.0%}**")

    # ── Trade history ────────────────────────────────────────────────
    st.subheader("Trade History")
    if sells:
        trade_df = pd.DataFrame(sells)
        trade_df["time"] = pd.to_datetime(trade_df["time"], format="mixed", utc=True).dt.tz_convert(CET).dt.strftime("%Y-%m-%d %H:%M")
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
                    win_df["time"] = pd.to_datetime(win_df["time"], format="mixed", utc=True).dt.tz_convert(CET).dt.strftime("%m-%d %H:%M")
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
                    loss_df["time"] = pd.to_datetime(loss_df["time"], format="mixed", utc=True).dt.tz_convert(CET).dt.strftime("%m-%d %H:%M")
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

    last_ts = sig_data.get("timestamp", "")
    if last_ts:
        last_cet = pd.to_datetime(last_ts, utc=True).tz_convert(CET).strftime("%Y-%m-%d %H:%M")
    else:
        last_cet = "?"
    st.caption(f"Last update: {last_cet} CET | CI/CD: every 5 min via GitHub Actions")

    if refresh_sec:
        time.sleep(refresh_sec)
        st.rerun()


if __name__ == "__main__":
    main()
