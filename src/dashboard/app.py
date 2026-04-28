"""
app.py — Streamlit real-time fraud monitoring dashboard.

Run with:  streamlit run src/dashboard/app.py --server.port 8501

Layout:
  ┌─────────────────────────────────────────┐
  │  Header + status bar                    │
  ├───────────┬───────────┬─────────────────┤
  │ Total tx  │  Flagged  │  Flag rate      │
  ├───────────┴───────────┴─────────────────┤
  │  Transaction volume chart (live)        │
  ├─────────────────────────────────────────┤
  │  Fraud score distribution               │
  ├──────────────────┬──────────────────────┤
  │  Drift over time │  Score heatmap       │
  ├──────────────────┴──────────────────────┤
  │  Live fraud alerts table                │
  └─────────────────────────────────────────┘
"""

import time
import sys
import os
import json
import logging

import boto3
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime

# Support both Docker (/app layout) and local dev (repo root layout)
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parent.parent.parent   # src/dashboard/ -> project root
for _p in ["/app", "/app/src", str(_root), str(_root / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import store
import kafka_reader

logging.basicConfig(level=logging.INFO)

# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fraud Detection Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
  }

  .stApp { background-color: #0a0e1a; }

  h1, h2, h3 { font-family: 'Space Mono', monospace; }

  .metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2236 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
  }

  .metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1;
    margin: 8px 0 4px;
  }

  .metric-label {
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748b;
  }

  .alert-row {
    background: rgba(239, 68, 68, 0.08);
    border-left: 3px solid #ef4444;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 4px 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
  }

  .status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s infinite;
  }

  .status-live { background: #22c55e; box-shadow: 0 0 8px #22c55e; }
  .status-off  { background: #6b7280; }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
  }

  section[data-testid="stSidebar"] { display: none; }

  div[data-testid="stHorizontalBlock"] > div { gap: 1rem; }

  .section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #475569;
    margin: 24px 0 12px;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 6px;
  }
</style>
""", unsafe_allow_html=True)

# ── init ──────────────────────────────────────────────────────────────────────

store.init_db()

# start (or restart) the background reader thread
kafka_reader.start()

REFRESH_INTERVAL = int(os.getenv("DASHBOARD_REFRESH_S", "3"))
FRAUD_THRESHOLD  = float(os.getenv("FRAUD_THRESHOLD", "0.5"))

# ── plotly theme ──────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#94a3b8", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor="#1e293b", zerolinecolor="#1e293b"),
    yaxis=dict(gridcolor="#1e293b", zerolinecolor="#1e293b"),
)

ACCENT   = "#3b82f6"   # blue
DANGER   = "#ef4444"   # red
WARNING  = "#f59e0b"   # amber
SUCCESS  = "#22c55e"   # green
MUTED    = "#334155"

# ── helpers ───────────────────────────────────────────────────────────────────

def _color_score(score: float) -> str:
    if score >= 0.8:  return DANGER
    if score >= 0.5:  return WARNING
    return SUCCESS


def render_header(status: dict):
    running = status.get("running", False)
    total   = status.get("total", 0)
    last_ts = status.get("last_ts") or "—"

    error   = status.get("error")
    dot_cls = "status-live" if running else "status-off"
    label   = "LIVE" if running else ("ERROR" if error else "OFFLINE")

    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center;
                padding: 0 0 20px 0; border-bottom: 1px solid #1e293b; margin-bottom: 24px;">
      <div>
        <h1 style="margin:0; font-size:1.6rem; color:#e2e8f0;">
          🛡️ Fraud Detection Monitor
        </h1>
        <p style="margin:4px 0 0; color:#475569; font-size:0.85rem;">
          BTC/USDT · XGBoost · SageMaker · Real-time
        </p>
      </div>
      <div style="text-align:right;">
        <span class="status-dot {dot_cls}"></span>
        <span style="font-family:'Space Mono',monospace; font-size:0.8rem; color:#94a3b8;">
          {label} · {total:,} scored · last {last_ts[:19] if last_ts != '—' else '—'}
          {f'<br><span style="color:#ef4444;font-size:0.7rem;">{error}</span>' if error else ''}
        </span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_kpi_cards(stats: dict, alerts: list):
    total   = int(stats.get("total") or 0)
    flagged = int(stats.get("flagged") or 0)
    rate    = flagged / total * 100 if total else 0
    avg_sc  = float(stats.get("avg_score") or 0)
    max_sc  = float(stats.get("max_score") or 0)
    avg_px  = float(stats.get("avg_price") or 0)

    cols = st.columns(5)
    cards = [
        ("Total Trades",     f"{total:,}",           ACCENT),
        ("Fraud Alerts",     f"{flagged:,}",          DANGER if flagged else SUCCESS),
        ("Flag Rate",        f"{rate:.2f}%",          WARNING if rate > 1 else SUCCESS),
        ("Avg Fraud Score",  f"{avg_sc:.4f}",         _color_score(avg_sc)),
        ("Avg BTC Price",    f"${avg_px:,.2f}",       ACCENT),
    ]
    for col, (label, value, color) in zip(cols, cards):
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value" style="color:{color};">{value}</div>
        </div>
        """, unsafe_allow_html=True)


def render_volume_chart(preds: list):
    if not preds:
        st.info("Waiting for trade data…")
        return

    df = pd.DataFrame(preds)
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts")

    # resample to 5-second buckets
    df = df.set_index("ts")
    vol   = df["fraud_score"].resample("5s").count().rename("total")
    fraud = df["is_fraud"].resample("5s").sum().rename("fraud")
    avg_s = df["fraud_score"].resample("5s").mean().rename("avg_score")

    combined = pd.concat([vol, fraud, avg_s], axis=1).dropna().reset_index()
    combined.columns = ["ts", "total", "fraud", "avg_score"]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=combined["ts"], y=combined["total"],
        name="Trades", marker_color=MUTED, opacity=0.7,
    ), secondary_y=False)

    fig.add_trace(go.Bar(
        x=combined["ts"], y=combined["fraud"],
        name="Flagged", marker_color=DANGER, opacity=0.9,
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=combined["ts"], y=combined["avg_score"],
        name="Avg Score", mode="lines",
        line=dict(color=WARNING, width=2),
    ), secondary_y=True)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Transaction Volume & Fraud Rate (5s buckets)",
        barmode="overlay",
        legend=dict(orientation="h", y=1.1),
        height=260,
    )
    fig.update_yaxes(title_text="Count",      secondary_y=False)
    fig.update_yaxes(title_text="Avg Score",  secondary_y=True,
                     range=[0, 1], gridcolor="rgba(0,0,0,0)")

    st.plotly_chart(fig, use_container_width=True)


def render_score_distribution(preds: list):
    if not preds:
        return
    df  = pd.DataFrame(preds)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df[df["is_fraud"] == 0]["fraud_score"],
        name="Legit", nbinsx=40,
        marker_color=ACCENT, opacity=0.7,
    ))
    fig.add_trace(go.Histogram(
        x=df[df["is_fraud"] == 1]["fraud_score"],
        name="Fraud", nbinsx=40,
        marker_color=DANGER, opacity=0.8,
    ))
    fig.add_vline(x=FRAUD_THRESHOLD, line_dash="dash", line_color=WARNING,
                  annotation_text=f"Threshold {FRAUD_THRESHOLD}")
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Fraud Score Distribution",
        barmode="overlay",
        legend=dict(orientation="h", y=1.1),
        height=260,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_drift_chart(drift_rows: list):
    if not drift_rows:
        st.info(f"Drift computed every {kafka_reader.DRIFT_WINDOW} predictions. Keep watching…")
        return

    df  = pd.DataFrame(drift_rows).sort_values("ts")
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=df["ts"], y=df["drift_score"],
        name="Drift Score", mode="lines+markers",
        line=dict(color=DANGER, width=2),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.1)",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=df["ts"], y=df["mean_score"],
        name="Window Mean", mode="lines",
        line=dict(color=WARNING, width=1.5, dash="dot"),
    ), secondary_y=True)

    fig.add_hrect(y0=1.0, y1=df["drift_score"].max() + 0.1 if len(df) else 2,
                  fillcolor="rgba(239,68,68,0.05)", line_width=0,
                  annotation_text="Drift Zone", annotation_position="top left")

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Data Drift Score Over Time",
        height=260,
        legend=dict(orientation="h", y=1.1),
    )
    fig.update_yaxes(title_text="Drift Score",   secondary_y=False)
    fig.update_yaxes(title_text="Window Mean",   secondary_y=True,
                     range=[0, 1], gridcolor="rgba(0,0,0,0)")

    st.plotly_chart(fig, use_container_width=True)


def render_score_heatmap(preds: list):
    """Score intensity over time as a heatmap (time × minute-of-hour)."""
    if len(preds) < 10:
        st.info("Building heatmap — need more data…")
        return

    df = pd.DataFrame(preds)
    df["ts"]     = pd.to_datetime(df["ts"])
    df["minute"] = df["ts"].dt.floor("1min")
    df["second"] = df["ts"].dt.second // 10 * 10   # 10s slots

    pivot = df.pivot_table(
        index="second", columns="minute",
        values="fraud_score", aggfunc="mean"
    ).fillna(0)

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(c)[-8:-3] for c in pivot.columns],
        y=[f"{r}s" for r in pivot.index],
        colorscale=[[0, "#0a0e1a"], [0.5, ACCENT], [1, DANGER]],
        showscale=True,
        colorbar=dict(title="Score", tickfont=dict(color="#94a3b8")),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Score Heatmap (10s slots × minute)",
        height=260,
    )
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data(ttl=300)
def fetch_model_info_from_s3() -> dict:
    """
    Reads current model metrics and recent retraining history from S3.
    Cached for 5 minutes so the dashboard doesn't hammer S3 on every rerun.
    Returns an empty dict if S3 is not configured or the files don't exist yet.
    """
    bucket = os.getenv("S3_BUCKET", "")
    region = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
    if not bucket:
        return {}
    try:
        s3 = boto3.client("s3", region_name=region)

        current: dict = {}
        try:
            obj = s3.get_object(Bucket=bucket, Key="models/current_model_metrics.json")
            current = json.loads(obj["Body"].read())
        except Exception:
            pass

        retraining_events: list[dict] = []
        try:
            paginator = s3.get_paginator("list_objects_v2")
            keys = []
            for page in paginator.paginate(Bucket=bucket, Prefix="logs/retraining_reports/"):
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])
            for key in sorted(keys)[-20:]:
                body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                retraining_events.append(json.loads(body))
        except Exception:
            pass

        return {"current": current, "retraining_events": retraining_events}
    except Exception:
        return {}


def render_model_performance(stats: dict, model_info: dict):
    """Renders the model performance panel with AUC, retrain history, and flag pie."""
    current = model_info.get("current", {})
    events  = model_info.get("retraining_events", [])

    if current.get("auc"):
        auc_color = SUCCESS if current["auc"] >= 0.90 else WARNING
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom:12px;">
          <div class="metric-label">Current Model AUC</div>
          <div class="metric-value" style="color:{auc_color};">{current['auc']:.4f}</div>
          <div class="metric-label" style="margin-top:4px;">
            deployed {current.get('deployed_at','')[:10]} &nbsp;·&nbsp;
            {current.get('job_name','')[:30]}
          </div>
        </div>
        """, unsafe_allow_html=True)

    if events:
        fig = go.Figure()
        xs = [e.get("completed_at", "")[:16] for e in events]
        labels = [
            f"by {e.get('triggered_by','manual')[:20]}"
            for e in events
        ]
        fig.add_trace(go.Scatter(
            x=xs, y=[1] * len(xs),
            mode="markers+text",
            marker=dict(size=12, color=WARNING, symbol="diamond"),
            text=labels,
            textposition="top center",
            textfont=dict(color="#94a3b8", size=9),
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title="Retraining Events",
            height=140,
            yaxis=dict(visible=False),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    total   = int(stats.get("total") or 0)
    flagged = int(stats.get("flagged") or 0)
    legit   = total - flagged
    if total:
        fig = go.Figure(go.Pie(
            labels=["Legitimate", "Flagged Fraud"],
            values=[legit, flagged],
            hole=0.65,
            marker=dict(colors=[ACCENT, DANGER]),
            textinfo="percent+label",
            textfont=dict(color="#e2e8f0"),
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=220,
            showlegend=False,
            annotations=[dict(
                text=f"{flagged/total*100:.1f}%<br>fraud",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color=DANGER, family="Space Mono"),
            )],
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Waiting for data…")


def render_alerts_table(alerts: list):
    if not alerts:
        st.success("✅ No fraud alerts yet.")
        return

    for a in alerts[:20]:
        score = float(a.get("fraud_score", 0))
        bar_w = int(score * 100)
        bar_c = DANGER if score >= 0.8 else WARNING
        st.markdown(f"""
        <div class="alert-row">
          <span style="color:#ef4444;">⚠ FRAUD</span>
          &nbsp;|&nbsp;
          <span style="color:#94a3b8;">{a.get('ts','')[:19]}</span>
          &nbsp;|&nbsp;
          <span style="color:#e2e8f0;">{a.get('symbol','')}</span>
          &nbsp;|&nbsp;
          Price <b>${float(a.get('price',0)):,.2f}</b>
          &nbsp;|&nbsp;
          Qty <b>{float(a.get('quantity',0)):.5f}</b>
          &nbsp;|&nbsp;
          Score
          <span style="color:{bar_c};"><b>{score:.4f}</b></span>
          <div style="background:{MUTED};border-radius:4px;height:4px;margin-top:6px;">
            <div style="background:{bar_c};width:{bar_w}%;height:4px;border-radius:4px;"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ── main render ───────────────────────────────────────────────────────────────

def main():
    status     = kafka_reader.get_status()
    stats      = store.fetch_summary_stats()
    preds      = store.fetch_recent_predictions(limit=500)
    alerts     = store.fetch_fraud_alerts(limit=20)
    drift_rows = store.fetch_drift_history(limit=200)
    model_info = fetch_model_info_from_s3()

    render_header(status)
    render_kpi_cards(stats, alerts)

    st.markdown('<div class="section-title">Transaction Volume</div>', unsafe_allow_html=True)
    render_volume_chart(preds)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-title">Score Distribution</div>', unsafe_allow_html=True)
        render_score_distribution(preds)
    with col_r:
        st.markdown('<div class="section-title">Drift Monitor</div>', unsafe_allow_html=True)
        render_drift_chart(drift_rows)

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        st.markdown('<div class="section-title">Score Heatmap</div>', unsafe_allow_html=True)
        render_score_heatmap(preds)
    with col_r2:
        st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
        render_model_performance(stats, model_info)

    st.markdown('<div class="section-title">Live Fraud Alerts</div>', unsafe_allow_html=True)
    render_alerts_table(alerts)

    # auto-refresh
    time.sleep(REFRESH_INTERVAL)
    st.rerun()


if __name__ == "__main__":
    main()


    