import streamlit as st
import numpy as np
import joblib
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()   # <-- THIS loads the .env file

# ===========================
# LOAD MODEL, SCALER, FEATURES
# ===========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

xgb = joblib.load(os.path.join(BASE_DIR, "xgb_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
feature_names = joblib.load(os.path.join(BASE_DIR, "feature_names.pkl"))

# ===========================
# GROQ CLIENT SETUP
# ===========================
# ❗ Do NOT hardcode your API key in code.
# Set it in your environment instead: export GROQ_API_KEY="your_key"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


def groq_maintenance_report(features_dict, prediction, prob):
    prompt = f"""
    You are a senior predictive maintenance engineer.

    Machine snapshot (feature → value):
    {features_dict}

    Model prediction:
    - Failure risk label (1 = high risk, 0 = low risk): {prediction}
    - Failure probability: {prob:.2f}

    Write a short report:
    - 1–2 sentences describing machine condition
    - Mention which signals (e.g., torque, speed, temperature, tool wear) look suspicious
    - 2–3 concrete maintenance recommendations
    - Simple English, no markdown, under 120 words.
    """

    response = client.responses.create(
        model="openai/gpt-oss-20b",
        input=prompt,
    )
    return response.output_text


# ===========================
# STREAMLIT UI
# ===========================

st.set_page_config(page_title="Predictive Maintenance Assistant", layout="wide")

st.title("🔧 Predictive Maintenance Assistant")

tab1, tab2 = st.tabs(["📈 Failure Risk Calculator", "💬 Maintenance Chatbot"])


# -------- TAB 1: Failure Risk Calculator --------
with tab1:
    st.subheader("📈 Failure Risk Calculator")

    # You can adjust these defaults and ranges to match your dataset
    col1, col2, col3 = st.columns(3)

    with col1:
        air_temp = st.number_input(
            "Air_temperature_(K)", min_value=250.0, max_value=350.0, value=300.0
        )
        process_temp = st.number_input(
            "Process_temperature_(K)", min_value=250.0, max_value=400.0, value=310.0
        )

    with col2:
        rotational_speed = st.number_input(
            "Rotational_speed_(rpm)", min_value=500.0, max_value=3000.0, value=1500.0
        )
        torque = st.number_input(
            "Torque_(Nm)", min_value=0.0, max_value=100.0, value=40.0
        )

    with col3:
        tool_wear = st.number_input(
            "Tool_wear_(min)", min_value=0.0, max_value=300.0, value=100.0
        )
        type_l = st.selectbox("Type_L", options=[0, 1], index=0)
        type_m = st.selectbox("Type_M", options=[0, 1], index=0)

    # Derived features (must match what you did in notebook)
    temp_delta = process_temp - air_temp
    power_est = torque * rotational_speed

    # Build feature vector in the SAME ORDER as training
    input_dict = {
        "Air_temperature_(K)": air_temp,
        "Process_temperature_(K)": process_temp,
        "Rotational_speed_(rpm)": rotational_speed,
        "Torque_(Nm)": torque,
        "Tool_wear_(min)": tool_wear,
        "Temp_delta": temp_delta,
        "Power_est": power_est,
        "Type_L": type_l,
        "Type_M": type_m,
    }

    # Align to full feature_names (if some columns exist that you are not manually entering, default to 0)
    x_vec = [float(input_dict.get(fname, 0.0)) for fname in feature_names]
    x_vec = np.array(x_vec).reshape(1, -1)

    # If your XGBoost in training used scaled data, use scaler here.
    # If not, you can remove scaler entirely.
    # Here I'll assume you trained XGB on UNscaled features (like we did earlier), so no scaling:
    # x_vec_scaled = scaler.transform(x_vec)

    if st.button("Predict Failure Risk", key="predict_btn"):
        prob = float(xgb.predict_proba(x_vec)[0, 1])
        pred = int(prob >= 0.5)

        st.subheader("📊 Model Output")
        st.write(f"**Failure probability:** {prob:.3f}")
        st.write(
            f"**Predicted label:** {'Failure (1)' if pred == 1 else 'Healthy (0)'}"
        )

        if GROQ_API_KEY:
            st.subheader("🤖 AI Maintenance Report")
            with st.spinner("Generating AI report..."):
                report = groq_maintenance_report(input_dict, pred, prob)
            st.write(report)
        else:
            st.warning("Set GROQ_API_KEY environment variable to enable AI report.")


# -------- TAB 2: Maintenance Chatbot --------
with tab2:
    st.subheader("💬 Predictive Maintenance Chatbot")
    st.write(
        "Ask questions about model behavior, important features, or maintenance strategy."
    )

    user_question = st.text_area(
        "Your question:",
        placeholder="e.g., Why is high torque dangerous? How can we reduce false alarms? What features matter most?",
    )

    if st.button("Ask RiskBot", key="ask_groq_btn"):
        if not GROQ_API_KEY:
            st.warning("Set GROQ_API_KEY to enable chatbot.")
        elif not user_question.strip():
            st.info("Please type a question first.")
        else:
            context = f"""
            We built a predictive maintenance model for industrial machines using features:
            {feature_names}

            The best model is XGBoost with ~82% recall and ~99% overall accuracy.
            Important features: torque, rotational speed, tool wear, temperature delta, power estimate.

            The system predicts machine failure (1) vs healthy (0).
            """
            prompt = f"""
            You are an expert in predictive maintenance and machine learning.

            Context:
            {context}

            User question:
            {user_question}

            Answer in 2–4 short paragraphs, simple language. Focus on practical, engineering-oriented advice.
            """

            response = client.responses.create(
                model="openai/gpt-oss-20b",
                input=prompt,
            )
            answer = response.output_text

            st.markdown("**Groq's answer:**")
            st.write(answer)
