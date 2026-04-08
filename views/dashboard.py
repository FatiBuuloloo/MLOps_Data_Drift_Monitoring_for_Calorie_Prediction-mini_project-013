import streamlit as st
import pickle, sqlite3, os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pkl_path = "calorie_prediction/baseline_distribution.pkl"
db_path  = "calorie_prediction/SQLite_calorie.db"
num_cols = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]
db_cols  = ["age", "height", "weight", "duration", "heart_rate", "body_temp"]
col_map  = dict(zip(db_cols, num_cols))
icon_map = {"Age": "🎂", "Height": "📏", "Weight": "⚖️","Duration": "⏱️", "Heart_Rate": "❤️", "Body_Temp": "🌡️"}

@st.cache_resource(show_spinner=False)
def load_baseline():
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def fetch_user_logs():
    if not os.path.exists(db_path):
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM user_logs ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def kl_divergence(p, q):
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    p = np.clip(p, 1e-10, None)
    q = np.clip(q, 1e-10, None)
    p /= p.sum()
    q /= q.sum()
    kl = float(np.sum(p * np.log(p / q)))
    return kl if np.isfinite(kl) else 0.0


def compute_kl_scores(baseline, df_live_transformed):
    scores = {}
    for feat in num_cols:
        dist          = baseline["distributions"][feat]
        bins          = dist["bins"]
        baseline_prob = dist["prob"]
        col_data = df_live_transformed[feat].dropna().values
        if len(col_data) < 5:
            scores[feat] = 0.0
            continue
        counts, _ = np.histogram(col_data, bins=bins, density=True)
        q_prob     = (counts + 1e-10) / np.sum(counts + 1e-10)
        scores[feat] = kl_divergence(baseline_prob, q_prob)
    return scores

def drift_color(score):
    if score < 0.1:   return "#4ade80", "Stable",   "🟢"
    elif score < 0.5: return "#facc15", "Moderate",  "🟡"
    else:             return "#f87171", "High Drift", "🔴"

def build_histogram_fig(baseline, df_live):
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f"{icon_map[c]} {c}" for c in num_cols],
        vertical_spacing=0.14, horizontal_spacing=0.08)
    positions = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
    for idx, feat in enumerate(num_cols):
        row, col = positions[idx]
        dist        = baseline["distributions"][feat]
        bins        = dist["bins"]
        b_prob      = dist["prob"]
        bin_centers = (bins[:-1] + bins[1:]) / 2
        fig.add_trace(go.Bar(
            x=bin_centers, y=b_prob,
            name="Baseline (P)" if idx == 0 else None,
            showlegend=(idx == 0),
            marker_color="rgba(56,189,248,0.45)",
            marker_line_color="#38bdf8", marker_line_width=1,
            legendgroup="baseline"
        ), row=row, col=col)
        if feat in df_live.columns and len(df_live) >= 5:
            col_data = df_live[feat].dropna().values
            counts, _ = np.histogram(col_data, bins=bins, density=True)
            q_prob    = (counts + 1e-10) / np.sum(counts + 1e-10)
            fig.add_trace(go.Bar(
                x=bin_centers, y=q_prob,
                name="Live Data (Q)" if idx == 0 else None,
                showlegend=(idx == 0),
                marker_color="rgba(249,115,22,0.45)",
                marker_line_color="#f97316", marker_line_width=1,
                legendgroup="live"
            ), row=row, col=col)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        barmode="overlay",
        height=460,
        margin=dict(t=50, b=20, l=20, r=20),
        font=dict(family="Inter", color="#cbd5e1"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="#1e293b", bordercolor="#334155", borderwidth=1
        )
    )
    fig.update_xaxes(showgrid=True, gridcolor="#334155", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#334155", zeroline=False)
    return fig

def build_kl_bar(scores):
    colors = [drift_color(v)[0] for v in scores.values()]
    fig = go.Figure(go.Bar(
        x=list(scores.keys()),
        y=list(scores.values()),
        marker_color=colors,
        marker_line_color="#334155",
        marker_line_width=1,
        text=[f"{v:.4f}" for v in scores.values()],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=11)))
    fig.add_hline(
        y=0.1, line_dash="dot", line_color="#facc15",
        annotation_text="Warning (0.1) ⚠️", annotation_font_color="#facc15")
    fig.add_hline(
        y=0.5, line_dash="dot", line_color="#f87171",
        annotation_text="Critical (0.5) 🚨", annotation_font_color="#f87171")
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        height=300,
        margin=dict(t=20, b=20, l=40, r=20),
        font=dict(family="Inter", color="#cbd5e1"),
        yaxis=dict(title="KL Divergence Score", gridcolor="#334155"),
        xaxis=dict(gridcolor="#334155"),
        showlegend=False)
    return fig

def render():
    st.markdown("""
    <style>
    #header-dash {
        background: linear-gradient(90deg,#f97316,#f43f5e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;}
    </style>
    <div style='padding:1.5rem 0 .5rem;'>
        <h1 id='header-dash' style='font-size:2rem; font-weight:700; margin:0;'>
            MLOps Dashboard
        </h1>
        <p style='color:#94a3b8; font-size:.95rem; margin:.4rem 0 0;'>
            Real-time data drift monitoring via KL Divergence —
            comparing live inference data against training baseline.
        </p>
    </div>
    <hr style='border-color:#334155; margin:.8rem 0 1.5rem;'/>
    """, unsafe_allow_html=True)
    try:
        baseline = load_baseline()
    except Exception as e:
        st.error(f"Failed to load baseline: {e}")
        return
    df_raw = fetch_user_logs()
    if df_raw.empty:
        st.info("No user logs yet. Go to **User Prediction** and submit some data first!")
        return
    df_live = (
        df_raw[db_cols]
        .rename(columns=col_map)
        .dropna()
        .reset_index(drop=True))
    kl_scores = compute_kl_scores(baseline, df_live)
    avg_kl = float(np.mean(list(kl_scores.values())))
    valid_scores = {k: v for k, v in kl_scores.items() if np.isfinite(v) and v > 0}
    if valid_scores:
        max_feat = max(valid_scores, key=valid_scores.get)
        max_kl   = valid_scores[max_feat]
    else:
        max_feat = "—"
        max_kl   = 0.0
    avg_c, avg_s, avg_icon = drift_color(avg_kl)
    _, _, max_icon          = drift_color(max_kl)
    overall = "STABLE 🟢" if avg_kl < 0.1 else "DEGRADING ⚠️" if avg_kl < 0.5 else "ALERT 🚨"
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("📥 Total Logs", f"{len(df_raw):,}")
    with c2:
        st.metric("Avg KL Divergence", f"{avg_kl:.4f}",
                  delta=f"{avg_icon} {avg_s}", delta_color="off")
    with c3:
        st.metric("Most Drifted Feature", max_feat,
                  delta=f"{max_icon} KL = {max_kl:.4f}", delta_color="off")
    with c4:
        st.metric("System Health", overall)
    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("""<div style='font-size:1rem; font-weight:600; color:#e2e8f0; margin-bottom:1rem;'>
        KL Divergence per Feature
    </div>
    """, unsafe_allow_html=True)
    card_cols = st.columns(6)
    for idx, feat in enumerate(num_cols):
        score = kl_scores[feat]
        color, status, icon = drift_color(score)
        with card_cols[idx]:
            st.markdown(f"""<div style='background:#1e293b; border:1.5px solid {color}33;
                        border-radius:12px; padding:1rem .8rem; text-align:center;
                        box-shadow:0 0 12px {color}22;'>
                <div style='font-size:1.3rem;'>{icon_map[feat]}</div>
                <div style='font-size:.78rem; color:#94a3b8; margin:.3rem 0;'>{feat}</div>
                <div style='font-size:1.35rem; font-weight:700; color:{color};'>
                    {score:.4f}
                </div>
                <div style='font-size:.7rem; color:{color}; margin-top:.25rem;
                            background:{color}22; border-radius:6px; padding:.15rem;'>
                    {icon} {status}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("""<div style='background:#1e293b; border:1px solid #334155; border-radius:16px;
                padding:1.2rem 1.5rem; margin-bottom:1.5rem;'>
        <div style='font-size:1rem; font-weight:600; color:#e2e8f0; margin-bottom:.8rem;'>
            KL Divergence Overview — All Features
        </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(build_kl_bar(kl_scores), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("""<div style='background:#1e293b; border:1px solid #334155; border-radius:16px;
                padding:1.2rem 1.5rem; margin-bottom:1.5rem;'>
        <div style='font-size:1rem; font-weight:600; color:#e2e8f0; margin-bottom:.3rem;'>
            Distribution Overlay — Baseline (P) vs Live Data (Q)
        </div>
        <div style='font-size:.82rem; color:#64748b; margin-bottom:.8rem;'>
            Both distributions are on the same Box-Cox transformed scale.
            Divergence indicates potential data drift.
        </div>
    """, unsafe_allow_html=True)
    st.plotly_chart(build_histogram_fig(baseline, df_live), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    df_trend = df_raw.head(50).copy()
    if "timestamp" in df_trend.columns and "predicted_calories" in df_trend.columns:
        df_trend["timestamp"] = pd.to_datetime(df_trend["timestamp"])
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=df_trend["timestamp"][::-1],
            y=df_trend["predicted_calories"][::-1],
            mode="lines+markers",
            line=dict(color="#38bdf8", width=2),
            marker=dict(color="#38bdf8", size=5),
            fill="tozeroy",
            fillcolor="rgba(56,189,248,0.08)",
            name="Predicted Calories"))
        fig_trend.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#1e293b",
            height=250,
            margin=dict(t=20, b=20, l=40, r=20),
            font=dict(family="Inter", color="#cbd5e1"),
            xaxis=dict(gridcolor="#334155"),
            yaxis=dict(title="Predicted (transformed scale)", gridcolor="#334155"),
            showlegend=False)
        st.markdown("""<div style='background:#1e293b; border:1px solid #334155; border-radius:16px;
                    padding:1.2rem 1.5rem; margin-bottom:1.5rem;'>
            <div style='font-size:1rem; font-weight:600; color:#e2e8f0; margin-bottom:.8rem;'>
                📈 Prediction Trend — Last 50 Logs
            </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("""<div style='font-size:1rem; font-weight:600; color:#e2e8f0; margin-bottom:.8rem;'>
        Latest 10 Incoming Logs (Raw Data Scale)
    </div>
    """, unsafe_allow_html=True)
    df_table = df_raw.head(10).copy()
    try:
        le = baseline["transformers"]["labelencoder"]
        pt = baseline["transformers"]["powertransformer"]
        mapping_for_pt = {
            "age": "Age", "height": "Height", "weight": "Weight",
            "duration": "Duration", "heart_rate": "Heart_Rate",
            "body_temp": "Body_Temp", "predicted_calories": "Calories"}
        df_to_inverse = df_table[list(mapping_for_pt.keys())].rename(columns=mapping_for_pt)
        inversed_array = pt.inverse_transform(df_to_inverse)
        df_table[list(mapping_for_pt.keys())] = inversed_array
        def decode_gender(g):
            try:
                return le.inverse_transform([int(g)])[0]
            except:
                return "Invalid Data"             
        df_table["gender"] = df_table["gender"].apply(decode_gender)
        display_df = df_table[[
            "timestamp", "gender", "age", "height", "weight",
            "duration", "heart_rate", "body_temp", "predicted_calories"]].copy()
        display_df.columns = [
            "Timestamp", "Gender", "Age", "Height(cm)",
            "Weight(kg)", "Duration(min)", "HR(bpm)", "Temp(°C)", "Calories(kcal)"]
        st.dataframe(
            display_df.style.format({
                "Age": "{:.0f}",
                "Height(cm)": "{:.1f}",
                "Weight(kg)": "{:.1f}",
                "Duration(min)": "{:.1f}",
                "HR(bpm)": "{:.0f}",
                "Temp(°C)": "{:.1f}",
                "Calories(kcal)": "{:.2f}"}),
            use_container_width=True,
            hide_index=True)
    except Exception as e:
        st.error(f"Error processing table data: {e}")
        st.dataframe(df_raw.head(10), use_container_width=True, hide_index=True)
    with st.expander("How to interpret drift scores"):
        st.markdown("""
        **KL Divergence** $KL(P \parallel Q) = \sum P(x) \cdot \log(P(x) / Q(x))$ measures how far the 
        live data distribution **Q** deviates from the training baseline distribution **P**.

        Both distributions are calculated on the **same Box-Cox scale** (no re-transformation is performed 
        here, the SQLite data is already stored in its transformed state by `prediction.py`).

        | Score | Status | Action |
        |---|---|---|
        | `< 0.1` | 🟢 **Stable** | No action required |
        | `0.1 – 0.5` | 🟡 **Moderate Drift** | Investigate data source; consider retraining |
        | `> 0.5` | 🔴 **High Drift** | Immediate retraining recommended |

        > A score of **0** indicates that the distribution is identical to the training baseline.
        """)

    st.markdown("<br/>", unsafe_allow_html=True)
    if st.button("🔄 Refresh Dashboard"):
        st.cache_resource.clear()
        st.rerun()