import streamlit as st

st.set_page_config(
    page_title="MLOps | Calorie Prediction",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded")

st.markdown("""
<style>
/* fonts & base */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;}
[data-testid="stSidebarNav"] label, 
[data-testid="stSidebar"] .stMarkdown div,
[data-testid="stSidebar"] .stRadio label {color: #e2e8f0 !important; }

/* main background */
[data-testid="stAppViewContainer"] { background: #0f172a; }
[data-testid="stHeader"] { background: transparent; }
.element-container [data-testid="stImage"] div button,
.element-container [data-testid="stImage"] div button:hover,
.element-container [data-testid="stImage"] div button:focus,
.element-container [data-testid="stImage"] div button:active {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
    width: 0 !important;
    height: 0 !important;}
div[data-baseweb="popover"] *, 
div[role="listbox"] *, 
div[data-baseweb="menu"] * {
    color: #000000 !important;}

/* global text */
h1, h2, h3, h4, p, label { color: #e2e8f0; }

/* metric cards */
[data-testid="stMetric"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem;}
[data-testid="stMetricValue"] { color: #38bdf8 !important; font-size: 1.6rem !important; }
[data-testid="stMetricLabel"] { color: #94a3b8 !important; }

/* dataframe */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* buttons */
.stButton > button {
    background: linear-gradient(135deg, #38bdf8, #6366f1);
    color: #fff !important;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.8rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all .3s;
    width: 100%;}
.stButton > button:hover { opacity: .85; transform: translateY(-1px); }

/* input widgets */
.stNumberInput input, 
div[data-baseweb="select"] > div {
    background-color: #1e293b !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;}

/* expander */
.streamlit-expanderHeader { background: #1e293b; border-radius: 8px; }
[data-testid="stExpanderDetails"] * {color: #e2e8f0 !important;}
code {
    color: #38bdf8 !important;
    background-color: #0f172a !important;
    padding: 0.2rem 0.4rem !important;
    border-radius: 4px !important;
    border: 1px solid #334155 !important;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""<div style='text-align:center; padding: 1.5rem 0 1rem;'>
        <div style='font-size:2.4rem;'>🔥</div>
        <div style='font-size:1.1rem; font-weight:700; color:#38bdf8; margin-top:.4rem;'>
            Calorie MLOps
        </div>
        <div style='font-size:.75rem; color:#64748b; margin-top:.2rem;'>
            Monitoring System
        </div>
    </div>
    <hr style='border-color:#334155; margin:.5rem 0 1.2rem;'/>
    """, unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["User Prediction", "MLOps Dashboard"],
        label_visibility="collapsed")
    st.markdown("""<hr style='border-color:#334155; margin:1.5rem 0 1rem;'/>
    <div style='font-size:.72rem; color:#475569; text-align:center;'>
        Model: <span style='color:#38bdf8;'>XGBoost Regressor</span><br/>
        Drift Metric: <span style='color:#38bdf8;'>KL Divergence</span><br/>
        DB: <span style='color:#38bdf8;'>SQLite (real-time)</span>
    </div>
    """, unsafe_allow_html=True)

if page == "User Prediction":
    from views.prediction import render
    render()
else:
    from views.dashboard import render
    render()