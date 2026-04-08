import streamlit as st
import pickle, sqlite3, datetime, os
import numpy as np
import pandas as pd

hf_base = "https://huggingface.co/spaces/Viewww/MLOps_Dashboard/resolve/main"
pkl_path = "calorie_prediction/baseline_distribution.pkl"
model_path = "calorie_prediction/xgb_model-calorie_prediction.pkl"
db_path = "calorie_prediction/SQLite_calorie.db"
num_cols = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]

@st.cache_resource(show_spinner=False)
def init_db():
    os.makedirs(os.path.dirname(db_path), exist_ok=True)    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS user_logs")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            gender INTEGER,
            age FLOAT,
            height FLOAT,
            weight FLOAT,
            duration FLOAT,
            heart_rate FLOAT,
            body_temp FLOAT,
            predicted_calories FLOAT)''')
    conn.commit()
    conn.close()
init_db()

@st.cache_resource(show_spinner=False)
def load_artifacts():
    with open(pkl_path, "rb") as f:
        baseline = pickle.load(f)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    pt = baseline["transformers"]["powertransformer"]
    le = baseline["transformers"]["labelencoder"]
    scaler = baseline["transformers"]["scaler"]
    return model, pt, le, scaler

def log_to_db(gender_enc, age, height, weight, duration, heart_rate, body_temp, predicted):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO user_logs 
        (gender, age, height, weight, duration, heart_rate, body_temp, predicted_calories)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (int(gender_enc), age, height, weight, duration, heart_rate, body_temp, predicted))
    conn.commit()
    conn.close()
    
def render():
    st.markdown("""
    <style>
    #header-pred {
        background: linear-gradient(90deg,#38bdf8,#818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;}
    [data-testid="stForm"] { border: none; padding: 0; }
    </style>
    <div style='padding:0.5rem 0 .5rem;'>
        <h1 id='header-pred' style='font-size:2rem; font-weight:700; margin:0;'>
            User Prediction
        </h1>
        <p style='color:#94a3b8; font-size:.95rem; margin:.4rem 0 0;'>
            Input your biometric and workout data — get an instant calorie prediction.
        </p>
    </div>
    <hr style='border-color:#334155; margin:.8rem 0 1.5rem;'/>
    """, unsafe_allow_html=True)
    try:
        model, pt, le, scaler = load_artifacts()
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}")
        return
    st.markdown("""
    <div style='background:#1e293b; border:1px solid #334155;
                border-radius:16px; padding:1.5rem 1.8rem 1rem; margin-bottom:1.5rem;'>
        <div style='font-size:1rem; font-weight:600; color:#38bdf8; margin-bottom:1rem;'>
            📋 Personal Information
        </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ["male", "female"], help="Biological sex")
    with col2:
        age = st.number_input("Age (years)", min_value=10, max_value=100, value=25, step=1)
    with col3:
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.5)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("""<div style='background:#1e293b; border:1px solid #334155;
                border-radius:16px; padding:1.5rem 1.8rem 1rem; margin-bottom:1.5rem;'>
        <div style='font-size:1rem; font-weight:600; color:#818cf8; margin-bottom:1rem;'>
            🏋️ Workout Parameters
        </div>
    """, unsafe_allow_html=True)
    col4, col5, col6, col7 = st.columns(4)
    with col4:
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=65.0, step=0.5)
    with col5:
        duration = st.number_input("Duration (min)", min_value=1.0, max_value=300.0, value=30.0, step=1.0)
    with col6:
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40.0, max_value=220.0, value=100.0, step=1.0)
    with col7:
        body_temp = st.number_input("Body Temp (°C)", min_value=35.0, max_value=42.5, value=38.5, step=0.1)
    st.markdown("</div>", unsafe_allow_html=True)
    predict_clicked = st.button("Predict Calories", use_container_width=True)
    if predict_clicked:
        try:
            bmi = weight / ((height / 100) ** 2)
            bmi_cat = ("Underweight" if bmi < 18.5
                        else "Normal" if bmi < 25
                        else "Overweight" if bmi < 30 else "Obese")
            bmi_color = {"Underweight": "#facc15", "Normal": "#4ade80",
                         "Overweight": "#fb923c", "Obese": "#f87171"}[bmi_cat]        
            st.markdown(f"""<div style='background:#0f172a; border:1px solid #334155; border-radius:10px;
                        padding:.8rem 1.2rem; margin-bottom:1.5rem; display:flex;
                        align-items:center; gap:1rem;'>
                <span style='color:#94a3b8; font-size:.88rem;'>Quick Stats:</span>
                <span style='color:#e2e8f0;'>BMI: <b style='color:{bmi_color};'>{bmi:.1f}</b>
                    <span style='font-size:.78rem; color:{bmi_color};'>({bmi_cat})</span>
                </span>
                <span style='color:#94a3b8;'>|</span>
                <span style='color:#e2e8f0;'>MHR Est.:
                    <b style='color:#38bdf8;'>{220 - age} bpm</b>
                </span>
                <span style='color:#94a3b8;'>|</span>
                <span style='color:#e2e8f0;'>HR Zone:
                    <b style='color:#818cf8;'>{int((heart_rate/(220-age))*100)}%</b>
                </span>
            </div>
            """, unsafe_allow_html=True)
            gender_enc = le.transform([gender])[0]
            num_values = [float(age), float(height), float(weight), float(duration), float(heart_rate), float(body_temp)]
            dummy_val = 1.0 
            full_cols = num_cols + ["Calories"]
            input_prep = pd.DataFrame([num_values + [dummy_val]], columns=full_cols)
            num_transformed = pt.transform(input_prep)[:, :6]
            age_pt, h_pt, w_pt, dur_pt, hr_pt, bt_pt = num_transformed[0]
            feature_names = ["Gender"] + num_cols
            X_combined = np.hstack([[gender_enc], num_transformed[0]]).reshape(1, -1)
            X_df = pd.DataFrame(X_combined, columns=feature_names)
            X_scaled = scaler.transform(X_df)
            val_transformed = float(model.predict(X_scaled)[0])
            dummy_features = np.ones((1, 6)) 
            padded_output = np.hstack((dummy_features, [[val_transformed]]))
            padded_df = pd.DataFrame(padded_output, columns=full_cols)
            inversed_result = pt.inverse_transform(padded_df)
            val_inversed = float(inversed_result[0, 6])
            log_to_db(gender_enc, age_pt, h_pt, w_pt, dur_pt, hr_pt, bt_pt, val_transformed)
            intensity = ("Low Intensity🟢" if val_inversed < 150
                         else "Moderate Intensity🟡" if val_inversed < 300
                         else "High Intensity🔴")
            st.markdown(f"""<div style='background: linear-gradient(135deg,#0f2744,#1a1040);
                        border:1px solid #3b82f6; border-radius:16px;
                        padding:2rem; text-align:center; margin-top:1.5rem;'>
                <div style='font-size:.9rem; color:#94a3b8;'>ESTIMATED CALORIES BURNED</div>
                <div style='font-size:3.5rem; font-weight:800; color:#38bdf8;'>{val_inversed:,.1f}</div>
                <div style='font-size:1rem; color:#94a3b8;'>kcal</div>
                <div style='margin-top:1rem; font-size:1rem; color:#f59e0b;'><b>{intensity}</b></div>
            </div>
            """, unsafe_allow_html=True)
            st.success("Analysis complete. Data logged to MLOps baseline.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    with st.expander("How does this work?"):
        st.markdown("""
        **Pipeline Steps:**
        1. **Input Capture** — User fills in biometric and workout features.
        2. **Label Encoding** — `Gender` is encoded using the same `LabelEncoder` fitted during training.
        3. **Box-Cox Transformation** — Numeric features are power-transformed via `PowerTransformer`to match the training distribution.
        4. **Feature Scaling** — A `StandardScaler` is applied to feed to the model.
        5. **Inference** — The XGBoost Regressor produces a calorie prediction.
        6. **Logging** — Transformed numeric input + prediction is written to SQLite for drift monitoring.
        """)