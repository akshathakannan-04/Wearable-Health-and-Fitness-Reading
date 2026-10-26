"""
Wearable Health Analytics & Fitness Readiness Dashboard
======================================================
Main Application: app.py

An interactive Streamlit web dashboard for analyzing wearable health data,
evaluating Fitness Readiness Scores (0-100), inspecting sleep architecture,
and interpreting machine learning models (Random Forest and XGBoost).
"""

import os
import tempfile
from datetime import datetime

# Configure Matplotlib writable directory before any sub-imports
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib

# Import pipeline helpers for automatic generation/training
from generate_data import (
    run_pipeline,
    engineer_features,
    calculate_readiness_score,
    XGBOOST_AVAILABLE
)

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Wearable Health & Fitness Readiness",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom modern CSS styling
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="metric-container"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 1rem 1.25rem;
        border-radius: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="metric-container"]:hover {
        border-color: #cbd5e1;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.08);
        transition: all 0.2s ease-in-out;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.825rem;
        font-weight: 600;
        border-radius: 9999px;
    }
    .badge-green { background-color: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }
    .badge-yellow { background-color: #fef9c3; color: #a16207; border: 1px solid #fef08a; }
    .badge-red { background-color: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }
    .disclaimer-card {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 0.85rem 1.15rem;
        border-radius: 0.375rem;
        margin-bottom: 1rem;
        font-size: 0.875rem;
        color: #92400e;
    }
    .recommendation-box {
        padding: 1.1rem 1.35rem;
        border-radius: 0.6rem;
        margin-bottom: 1.25rem;
        border: 1px solid transparent;
    }
    .rec-high { background-color: #ecfdf5; border-color: #a7f3d0; color: #065f46; }
    .rec-med { background-color: #eff6ff; border-color: #bfdbfe; color: #1e40af; }
    .rec-low { background-color: #fff7ed; border-color: #fed7aa; color: #9a3412; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# DATA & MODEL LOADING (CACHED)
# ==============================================================================

CSV_FILE = "wearable_data.csv"
MODEL_FILE = "trained_model.pkl"
PLOT_FILE = "feature_importance.png"


@st.cache_data(show_spinner=False)
def load_data(filepath: str) -> pd.DataFrame:
    """Loads and formats wearable data."""
    if not os.path.exists(filepath):
        return None
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_resource(show_spinner=False)
def load_trained_model(filepath: str):
    """Loads trained machine learning model payload."""
    if not os.path.exists(filepath):
        return None
    return joblib.load(filepath)


# Auto-generate data and train model if required files are absent
if not os.path.exists(CSV_FILE) or not os.path.exists(MODEL_FILE):
    with st.spinner("Initializing system: Generating synthetic wearable dataset & training ML models..."):
        run_pipeline(
            csv_path=CSV_FILE,
            model_path=MODEL_FILE,
            plot_path=PLOT_FILE,
            num_users=50,
            num_days=60
        )
    st.success("Dataset & predictive models initialized successfully!")

df_all = load_data(CSV_FILE)
model_payload = load_trained_model(MODEL_FILE)

if df_all is None or model_payload is None:
    st.error("Unable to load data or model files. Please verify generate_data.py execution.")
    st.stop()


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
st.sidebar.title("🩺 Health Analytics Hub")
st.sidebar.markdown("Personalized Wearable Biomarkers & Fitness Readiness")
st.sidebar.divider()

# User Selector
user_list = sorted(df_all["user_id"].unique())
selected_user = st.sidebar.selectbox("👤 Select User Profile", user_list, index=0)

# User-specific DataFrame
df_user_full = df_all[df_all["user_id"] == selected_user].sort_values(by="date").reset_index(drop=True)

# Date Range Filter
min_dt = df_user_full["date"].min().date()
max_dt = df_user_full["date"].max().date()

st.sidebar.markdown("📅 **Filter Date Window**")
date_range = st.sidebar.date_input(
    "Observation Period",
    value=(min_dt, max_dt),
    min_value=min_dt,
    max_value=max_dt
)

# Handle single date selection gracefully
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_dt, max_dt

# Filter user data
mask = (df_user_full["date"].dt.date >= start_date) & (df_user_full["date"].dt.date <= end_date)
df_filtered = df_user_full.loc[mask].copy()

if df_filtered.empty:
    st.sidebar.warning("No records found in selected date range. Reverting to full user history.")
    df_filtered = df_user_full.copy()

# Sidebar User Quick Summary
st.sidebar.divider()
st.sidebar.markdown(f"**Profile Summary: `{selected_user}`**")
st.sidebar.write(f"- Total Days Tracked: **{len(df_filtered)}**")
st.sidebar.write(f"- Avg Daily Steps: **{int(df_filtered['total_steps'].mean()):,}**")
st.sidebar.write(f"- Avg Sleep: **{df_filtered['sleep_duration_hours'].mean():.1f} hrs**")
st.sidebar.write(f"- Avg RHR: **{df_filtered['resting_heart_rate'].mean():.1f} bpm**")

st.sidebar.divider()
st.sidebar.caption("⚡ Wearable Health Analytics Platform\nv2.4 • Built with Python & Streamlit")


# ==============================================================================
# MAIN PAGE HEADER & DISCLAIMER
# ==============================================================================
st.title("⚡ Wearable Health Analytics & Fitness Readiness Dashboard")
st.markdown("Track physiological recovery, evaluate daily strain, and optimize athletic training readiness.")

# Prominent Educational Disclaimer
st.markdown("""
<div class="disclaimer-card">
    <strong>⚠️ Educational & Illustrative Notice:</strong> The Fitness Readiness Score and analytical metrics
    presented here are for educational, demonstrative, and software engineering purposes only. They do not constitute
    medical advice, clinical diagnostic evaluations, or validated determinations of autonomic burnout or physical illness.
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# LATEST RECORD METRICS & ACTIONABLE RECOMMENDATION
# ==============================================================================

# Accurately identify and label the latest available record
latest_record = df_filtered.iloc[-1]
latest_date_str = latest_record["date"].strftime("%B %d, %Y")

# Compute deltas against user's 7-day average or baseline
readiness_val = latest_record["Readiness_Score"]
rhr_val = latest_record["resting_heart_rate"]
rhr_dev = latest_record["rhr_14d_baseline_dev"]
steps_val = latest_record["total_steps"]
steps_7d = latest_record["steps_7d_avg"]
steps_delta = int(steps_val - steps_7d)
sleep_val = latest_record["sleep_duration_hours"]
rem_val = latest_record["rem_sleep_hours"]
deep_val = latest_record["deep_sleep_hours"]

st.markdown(f"### 📊 Latest Telemetry Overview &nbsp; `<small style='color:#64748b; font-size:0.95rem; font-weight:normal;'>Recorded Date: <b>{latest_date_str}</b></small>`", unsafe_allow_html=True)

# Four Key Metric Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    readiness_delta = round(readiness_val - df_filtered["Readiness_Score"].mean(), 1)
    st.metric(
        label="Readiness Score",
        value=f"{readiness_val:.0f} / 100",
        delta=f"{readiness_delta:+.1f} vs avg",
        help="Weighted balance: Sleep (35%), Activity (25%), RHR Baseline (25%), Stress (15%)."
    )

with col2:
    # Negative delta in RHR is desirable (lower resting heart rate = better recovery)
    st.metric(
        label="Resting Heart Rate",
        value=f"{rhr_val:.1f} bpm",
        delta=f"{rhr_dev:+.1f} bpm vs 14d base",
        delta_color="inverse",
        help="Resting heart rate relative to strictly prior 14-day rolling baseline."
    )

with col3:
    st.metric(
        label="Daily Steps",
        value=f"{steps_val:,}",
        delta=f"{steps_delta:+,} vs 7d avg",
        help="Total steps recorded on latest date compared to 7-day rolling average."
    )

with col4:
    restorative_pct = int(((rem_val + deep_val) / sleep_val) * 100) if sleep_val > 0 else 0
    st.metric(
        label="Sleep Duration",
        value=f"{sleep_val:.1f} hrs",
        delta=f"{restorative_pct}% Restorative (REM+Deep)",
        delta_color="off",
        help="Total sleep duration with percentage composed of REM and Deep sleep stages."
    )

# Dynamic Recommendation Banner based on Readiness Score
if readiness_val < 40:
    rec_class = "rec-low"
    rec_title = "🛑 Recommendation: Consider Rest or Light Active Recovery"
    rec_body = (
        f"Your latest Readiness Score is <b>{readiness_val:.0f}</b> (Below 40). "
        "Elevated resting heart rate, high stress, or accumulated sleep deficit suggest higher physiological strain. "
        "Prioritize sleep hygiene, hydration, mobility drills, or gentle walking rather than strenuous workouts."
    )
elif readiness_val <= 70:
    rec_class = "rec-med"
    rec_title = "⚖️ Recommendation: Balanced Activity & Moderate Routine"
    rec_body = (
        f"Your latest Readiness Score is <b>{readiness_val:.0f}</b> (40–70). "
        "Your physiological biomarkers reflect stable baseline recovery. "
        "You are well-suited for moderate cardiovascular training, steady-state runs, or standard strength sessions."
    )
else:
    rec_class = "rec-high"
    rec_title = "⚡ Recommendation: Prime Fitness Condition — Follow Planned Workout"
    rec_body = (
        f"Your latest Readiness Score is <b>{readiness_val:.0f}</b> (Above 70). "
        "Your sleep architecture, heart rate stability, and activity balance indicate strong recovery capacity. "
        "Feel confident tackling high-intensity workouts, progressive overload, or peak performance challenges if feeling well."
    )

st.markdown(f"""
<div class="recommendation-box {rec_class}">
    <h4 style="margin: 0 0 0.35rem 0; font-size: 1.05rem;">{rec_title}</h4>
    <p style="margin: 0; font-size: 0.92rem; line-height: 1.45;">{rec_body}</p>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# INTERACTIVE DATA VISUALIZATIONS (PLOTLY)
# ==============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Readiness & Activity Trends",
    "🌙 Sleep Architecture & Consistency",
    "🤖 Machine Learning & Feature Importance",
    "📋 Data Explorer & Export"
])

# ------------------------------------------------------------------------------
# TAB 1: READINESS & ACTIVITY TRENDS (DUAL-AXIS)
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Daily Fitness Readiness vs. Physical Activity")
    st.caption("Inspect how daily step counts and 7-day rolling activity interact with your readiness recovery curve.")

    fig_trends = go.Figure()

    # Steps bars on secondary y-axis
    fig_trends.add_trace(go.Bar(
        x=df_filtered["date"],
        y=df_filtered["total_steps"],
        name="Daily Steps",
        marker_color="rgba(148, 163, 184, 0.45)",
        yaxis="y2",
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Steps:</b> %{y:,}<extra></extra>"
    ))

    # 7-day rolling steps line on secondary y-axis
    fig_trends.add_trace(go.Scatter(
        x=df_filtered["date"],
        y=df_filtered["steps_7d_avg"],
        name="7-Day Steps Avg",
        line=dict(color="#64748b", width=2, dash="dot"),
        yaxis="y2",
        hovertemplate="<b>7d Avg Steps:</b> %{y:,.0f}<extra></extra>"
    ))

    # Readiness Score line on primary y-axis
    fig_trends.add_trace(go.Scatter(
        x=df_filtered["date"],
        y=df_filtered["Readiness_Score"],
        name="Readiness Score",
        line=dict(color="#2563eb", width=3),
        mode="lines+markers",
        marker=dict(size=6, color="#1d4ed8"),
        yaxis="y1",
        hovertemplate="<b>Readiness:</b> %{y:.1f} / 100<extra></extra>"
    ))

    # Reference threshold lines for Readiness (70 and 40)
    fig_trends.add_hline(
        y=70, line_dash="dash", line_color="#16a34a", opacity=0.7,
        annotation_text="High Readiness (>70)", annotation_position="top left",
        annotation_font_size=10, annotation_font_color="#16a34a"
    )
    fig_trends.add_hline(
        y=40, line_dash="dash", line_color="#dc2626", opacity=0.7,
        annotation_text="Rest / Recovery Zone (<40)", annotation_position="bottom left",
        annotation_font_size=10, annotation_font_color="#dc2626"
    )

    fig_trends.update_layout(
        height=450,
        margin=dict(l=40, r=40, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title="Date", showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(
            title="Fitness Readiness Score (0 - 100)",
            range=[0, 105],
            showgrid=True,
            gridcolor="#f1f5f9"
        ),
        yaxis2=dict(
            title="Total Daily Steps",
            overlaying="y",
            side="right",
            showgrid=False,
            range=[0, max(df_filtered["total_steps"].max() * 1.25, 15000)]
        ),
        hovermode="x unified",
        plot_bgcolor="white"
    )

    st.plotly_chart(fig_trends, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 2: SLEEP ARCHITECTURE & CONSISTENCY
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Sleep Architecture Breakdown & Schedule Consistency")
    st.caption("Restorative sleep (REM and Deep) rebuilds cognitive and muscular systems. High rolling std reflects schedule instability.")

    col_s1, col_s2 = st.columns([2, 1])

    with col_s1:
        # Stacked breakdown of sleep components
        df_sleep_plot = df_filtered.copy()
        # Light sleep is remaining sleep duration
        df_sleep_plot["light_sleep_hours"] = np.maximum(
            0.0,
            df_sleep_plot["sleep_duration_hours"] - (df_sleep_plot["deep_sleep_hours"] + df_sleep_plot["rem_sleep_hours"])
        ).round(2)

        fig_sleep = go.Figure()

        fig_sleep.add_trace(go.Bar(
            x=df_sleep_plot["date"],
            y=df_sleep_plot["deep_sleep_hours"],
            name="Deep Sleep",
            marker_color="#1e3a8a",
            hovertemplate="Deep: %{y:.2f} hrs<extra></extra>"
        ))
        fig_sleep.add_trace(go.Bar(
            x=df_sleep_plot["date"],
            y=df_sleep_plot["rem_sleep_hours"],
            name="REM Sleep",
            marker_color="#3b82f6",
            hovertemplate="REM: %{y:.2f} hrs<extra></extra>"
        ))
        fig_sleep.add_trace(go.Bar(
            x=df_sleep_plot["date"],
            y=df_sleep_plot["light_sleep_hours"],
            name="Light Sleep",
            marker_color="#93c5fd",
            hovertemplate="Light: %{y:.2f} hrs<extra></extra>"
        ))

        # Target line (8.0 hours)
        fig_sleep.add_hline(
            y=8.0, line_dash="dot", line_color="#059669",
            annotation_text="Optimal Target (8.0h)", annotation_position="top right",
            annotation_font_size=10, annotation_font_color="#059669"
        )

        fig_sleep.update_layout(
            barmode="stack",
            height=400,
            title="Sleep Stages Breakdown (Hours)",
            margin=dict(l=40, r=20, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(title="Date"),
            yaxis=dict(title="Sleep Hours", range=[0, 12]),
            plot_bgcolor="white"
        )
        st.plotly_chart(fig_sleep, use_container_width=True)

    with col_s2:
        # 5-day Sleep Consistency Plot
        fig_consistency = px.line(
            df_filtered,
            x="date",
            y="sleep_5d_std",
            title="5-Day Sleep Consistency (Rolling Std Dev)",
            labels={"sleep_5d_std": "Rolling Std (Hours)", "date": "Date"},
            markers=True
        )
        fig_consistency.update_traces(line_color="#8b5cf6")
        fig_consistency.add_hline(
            y=1.0, line_dash="dash", line_color="#ef4444",
            annotation_text="High Irregularity (>1.0h)",
            annotation_position="top left",
            annotation_font_size=10, annotation_font_color="#ef4444"
        )
        fig_consistency.update_layout(
            height=400,
            margin=dict(l=30, r=20, t=40, b=40),
            plot_bgcolor="white",
            yaxis=dict(range=[0, max(df_filtered["sleep_5d_std"].max() * 1.3, 1.8)])
        )
        st.plotly_chart(fig_consistency, use_container_width=True)


# ------------------------------------------------------------------------------
# TAB 3: MACHINE LEARNING & FEATURE IMPORTANCE
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Machine Learning Performance & Interpretability")
    st.caption("Comparing Random Forest and XGBoost Regressors on a chronological test set.")

    # Model Evaluation Metrics Comparison Table
    m_data = model_payload.get("all_metrics", {})
    rf_m = m_data.get("Random Forest", {})
    xgb_m = m_data.get("XGBoost", {})

    comp_rows = [
        {
            "Model Architecture": "Random Forest Regressor",
            "RMSE (Root Mean Sq Error)": f"{rf_m.get('rmse', 0.0):.3f}",
            "MAE (Mean Absolute Error)": f"{rf_m.get('mae', 0.0):.3f}",
            "R² Score (Variance Explained)": f"{rf_m.get('r2', 0.0):.4f}",
            "Status": "Evaluated"
        }
    ]
    if xgb_m:
        comp_rows.append({
            "Model Architecture": "XGBoost Regressor",
            "RMSE (Root Mean Sq Error)": f"{xgb_m.get('rmse', 0.0):.3f}",
            "MAE (Mean Absolute Error)": f"{xgb_m.get('mae', 0.0):.3f}",
            "R² Score (Variance Explained)": f"{xgb_m.get('r2', 0.0):.4f}",
            "Status": "Best Model" if model_payload.get("model_name") == "XGBoost Regressor" else "Evaluated"
        })

    st.table(pd.DataFrame(comp_rows))

    col_m1, col_m2 = st.columns([1.1, 0.9])

    with col_m1:
        st.markdown(f"**Feature Importance Plot (`{model_payload.get('model_name')}`)**")
        if os.path.exists(PLOT_FILE):
            st.image(PLOT_FILE, caption="Relative Feature Importance for Readiness Score Estimation", use_container_width=True)
        else:
            st.info("Feature importance graphic not found; rerun pipeline.")

    with col_m2:
        st.markdown("**Methodological Disclaimer on Target Evaluation**")
        st.markdown("""
        > **Predictive Evaluation Note:**
        > The models were trained and tested using a strict **chronological train/test split**
        > (older days for training, unseen recent days for testing).
        >
        > Importantly, **variables that directly reveal the arithmetic target formula were withheld**.
        > The regressors learned feature representations directly from wearable sensors
        > (steps, active calories, sleep hours, REM, deep sleep, resting heart rate, stress, and rolling trends).
        >
        > High predictive performance on a synthetic or formula-derived target reflects the model's
        > capability to approximate non-linear surrogate functions and interactions.
        > **It does not establish real-world clinical health prediction accuracy.**
        """)

        st.divider()
        st.markdown("**⚡ Interactive Readiness Estimator (What-If Analysis)**")
        with st.expander("Try Interactive Prediction", expanded=False):
            w_sleep = st.slider("Sleep Duration (Hours)", 4.0, 10.0, float(latest_record["sleep_duration_hours"]), 0.1)
            w_deep = st.slider("Deep Sleep (Hours)", 0.5, 2.5, float(latest_record["deep_sleep_hours"]), 0.1)
            w_rem = st.slider("REM Sleep (Hours)", 0.5, 2.5, float(latest_record["rem_sleep_hours"]), 0.1)
            w_rhr = st.slider("Resting Heart Rate (bpm)", 45, 95, int(latest_record["resting_heart_rate"]), 1)
            w_stress = st.slider("Stress Score (0-100)", 10, 95, int(latest_record["stress_score"]), 1)
            w_steps = st.slider("Total Daily Steps", 2000, 25000, int(latest_record["total_steps"]), 500)

            # Assemble features vector
            feat_names = model_payload.get("feature_names", [])
            input_dict = {
                "total_steps": w_steps,
                "active_calories": int(w_steps * 0.045),
                "sleep_duration_hours": w_sleep,
                "rem_sleep_hours": w_rem,
                "deep_sleep_hours": w_deep,
                "resting_heart_rate": w_rhr,
                "stress_score": w_stress,
                "steps_7d_avg": latest_record["steps_7d_avg"],
                "calories_7d_avg": latest_record["calories_7d_avg"],
                "sleep_5d_std": latest_record["sleep_5d_std"]
            }
            input_df = pd.DataFrame([input_dict])[feat_names]
            active_model = model_payload["model"]
            pred_score = np.clip(active_model.predict(input_df)[0], 0, 100)

            st.metric(
                label="Model Predicted Readiness",
                value=f"{pred_score:.1f} / 100",
                delta=f"{pred_score - latest_record['Readiness_Score']:+.1f} vs latest recorded"
            )


# ------------------------------------------------------------------------------
# TAB 4: DATA EXPLORER & CSV EXPORT
# ------------------------------------------------------------------------------
with tab4:
    st.subheader(f"Recent Wearable Records for {selected_user}")
    st.caption("Showing filtered timeline records. Download the dataset below for external analysis or modeling.")

    # Formatted display DataFrame
    display_df = df_filtered[[
        "date",
        "Readiness_Score",
        "total_steps",
        "active_calories",
        "sleep_duration_hours",
        "deep_sleep_hours",
        "rem_sleep_hours",
        "resting_heart_rate",
        "rhr_14d_baseline_dev",
        "stress_score"
    ]].sort_values(by="date", ascending=False).reset_index(drop=True)

    # Format date string for display
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")

    st.dataframe(
        display_df,
        column_config={
            "date": "Date",
            "Readiness_Score": st.column_config.NumberColumn("Readiness Score", format="%.1f"),
            "total_steps": st.column_config.NumberColumn("Steps", format="%d"),
            "active_calories": st.column_config.NumberColumn("Active Cals", format="%d kcal"),
            "sleep_duration_hours": st.column_config.NumberColumn("Total Sleep", format="%.2f h"),
            "deep_sleep_hours": st.column_config.NumberColumn("Deep Sleep", format="%.2f h"),
            "rem_sleep_hours": st.column_config.NumberColumn("REM Sleep", format="%.2f h"),
            "resting_heart_rate": st.column_config.NumberColumn("RHR", format="%.1f bpm"),
            "rhr_14d_baseline_dev": st.column_config.NumberColumn("RHR Dev", format="%+.2f bpm"),
            "stress_score": st.column_config.NumberColumn("Stress Score", format="%d")
        },
        use_container_width=True,
        hide_index=True
    )

    # Download Button
    csv_bytes = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"⬇️ Download Filtered Data for {selected_user} (CSV)",
        data=csv_bytes,
        file_name=f"{selected_user}_wearable_data.csv",
        mime="text/csv"
    )

st.divider()
st.caption(
    "Disclaimer: This application was engineered as a reference Data Science and Machine Learning ecosystem project. "
    "All synthetic physiological biomarkers, readiness scores, and training suggestions are for educational simulation only."
)

