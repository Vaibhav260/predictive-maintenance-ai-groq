import streamlit as st
import numpy as np
import joblib
from openai import OpenAI
import os
import pandas as pd
from dotenv import load_dotenv
import time
import altair as alt
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

tab1, tab2, tab3 = st.tabs(
    ["📈 Failure Risk Calculator", "💬 Maintenance Chatbot", "📡 Live Monitoring"]
)


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
# -------- TAB 3: Live Monitoring (Simulation) --------
# -------- TAB 3: Live Monitoring (Simulation) --------
with tab3:
    # Small CSS tweak to keep spacing & avoid overlaps
    st.markdown(
        """
        <style>
        div[data-testid="stVerticalBlock"] > div {
            margin-bottom: 1.5rem !important;
        }
        .stAltairChart {
            min-height: 280px !important;
            margin-bottom: 1rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Real-time risk monitoring (simulated)</div>',
        unsafe_allow_html=True,
    )
    st.write(
        "This simulates a single machine streaming sensor data. At each step, "
        "we generate synthetic sensor readings, run the XGBoost model, and plot the failure risk over time."
    )

    # ---------- SESSION STATE INIT ----------
    if "live_data" not in st.session_state:
        st.session_state["live_data"] = pd.DataFrame(
            columns=[
                "time",
                "Air_temperature_(K)",
                "Process_temperature_(K)",
                "Rotational_speed_(rpm)",
                "Torque_(Nm)",
                "Tool_wear_(min)",
                "Temp_delta",
                "Power_est",
                "failure_prob",
            ]
        )
    if "events" not in st.session_state:
        st.session_state["events"] = []  # {"time", "failure_prob", "type"}
    if "sim_load" not in st.session_state:
        st.session_state["sim_load"] = 0.3
    if "sim_wear" not in st.session_state:
        st.session_state["sim_wear"] = 50.0
    if "sim_running" not in st.session_state:
        st.session_state["sim_running"] = False

    risk_threshold = 0.6
    critical_threshold = 0.8

    # ---------- CONTROLS ----------
    scenario = st.selectbox(
        "Load scenario",
        ["Normal operation", "Increasing load", "High stress", "Random fluctuation"],
        help="Preset patterns for the simulated machine load.",
    )

    col_controls = st.columns(3)
    with col_controls[0]:
        n_steps = st.number_input(
            "Steps to simulate",
            min_value=20,
            max_value=300,
            value=80,
            step=20,
            help="How many time steps to simulate in this run.",
        )
    with col_controls[1]:
        delay = st.number_input(
            "Delay between steps (sec)",
            min_value=0.1,
            max_value=3.0,
            value=0.4,
            step=0.1,
            help="How fast the live chart updates.",
        )
    with col_controls[2]:
        if st.button("Clear history 🗑️"):
            st.session_state["live_data"] = st.session_state["live_data"].iloc[0:0]
            st.session_state["events"] = []
            st.session_state["sim_load"] = 0.3
            st.session_state["sim_wear"] = 50.0

    # Start / Stop controls
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        start_btn = st.button(
            "Start simulation ▶️", disabled=st.session_state["sim_running"]
        )
    with col_btn2:
        stop_btn = st.button(
            "Stop simulation ⏸️", disabled=not st.session_state["sim_running"]
        )

    if start_btn:
        st.session_state["sim_running"] = True
    if stop_btn:
        st.session_state["sim_running"] = False
        st.rerun()

    # ---------- LAYOUT: LEFT (charts) / RIGHT (status & events) ----------
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.markdown("#### Failure probability over time")
        risk_chart_placeholder = st.empty()
        st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

        st.markdown("#### Key sensor trends")
        sensors_chart_placeholder = st.empty()

    with right_col:
        st.markdown("#### Current status")
        status_placeholder = st.empty()

        st.markdown("#### Session summary")
        summary_placeholder = st.empty()

        st.markdown("#### High-risk / anomaly events")
        events_placeholder = st.empty()

        if len(st.session_state["live_data"]) > 0:
            csv = st.session_state["live_data"].to_csv(index=False)
            st.download_button(
                "Download monitoring data 📥",
                csv,
                "live_monitoring.csv",
                "text/csv",
                key="download-csv",
            )

    # Progress bar
    progress_bar = st.progress(0.0)

    # ---------- SIMULATION LOOP ----------
    if st.session_state["sim_running"]:
        for step_idx in range(int(n_steps)):
            # Check if user requested stop
            if not st.session_state["sim_running"]:
                break

            # ---- 1. Update synthetic machine state (load & wear) ----
            load = st.session_state["sim_load"]

            if scenario == "Normal operation":
                load += np.random.normal(loc=0.0, scale=0.03)
            elif scenario == "Increasing load":
                load += 0.01 + np.random.normal(0.0, 0.02)
            elif scenario == "High stress":
                load = np.random.uniform(0.7, 1.0)
            else:  # Random fluctuation
                load = np.random.uniform(0.0, 1.0)

            load = float(np.clip(load, 0.0, 1.0))
            st.session_state["sim_load"] = load

            wear = st.session_state["sim_wear"] + np.random.uniform(0.2, 0.8)
            wear = float(np.clip(wear, 0.0, 300.0))
            st.session_state["sim_wear"] = wear

            base_air = 295.0
            base_speed = 1200.0
            base_torque = 30.0

            air_temp = np.random.normal(loc=base_air, scale=1.5)
            process_temp = air_temp + 5 + 20 * load + np.random.normal(0, 0.7)
            rotational_speed = base_speed + 1000 * load + np.random.normal(0, 80)
            torque = base_torque + 45 * load + np.random.normal(0, 5)
            tool_wear = wear

            temp_delta = process_temp - air_temp
            power_est = torque * rotational_speed

            feat_dict = {
                "Air_temperature_(K)": air_temp,
                "Process_temperature_(K)": process_temp,
                "Rotational_speed_(rpm)": rotational_speed,
                "Torque_(Nm)": torque,
                "Tool_wear_(min)": tool_wear,
                "Temp_delta": temp_delta,
                "Power_est": power_est,
                "Type_L": 0.0,
                "Type_M": 0.0,
            }

            # ---- 2. Predict risk ----
            x_vec_live = [float(feat_dict.get(fname, 0.0)) for fname in feature_names]
            x_vec_live = np.array(x_vec_live).reshape(1, -1)
            prob_live = float(xgb.predict_proba(x_vec_live)[0, 1])

            # ---- 3. Append to live_data ----
            new_row = {
                "time": len(st.session_state["live_data"]) + 1,
                "Air_temperature_(K)": air_temp,
                "Process_temperature_(K)": process_temp,
                "Rotational_speed_(rpm)": rotational_speed,
                "Torque_(Nm)": torque,
                "Tool_wear_(min)": tool_wear,
                "Temp_delta": temp_delta,
                "Power_est": power_est,
                "failure_prob": prob_live,
            }
            st.session_state["live_data"] = pd.concat(
                [st.session_state["live_data"], pd.DataFrame([new_row])],
                ignore_index=True,
            )

            df_plot = st.session_state["live_data"].copy()
            df_plot = df_plot.set_index("time")
            df_last = df_plot.tail(100).reset_index()

            # ---- 4. Risk chart with threshold line + coloring ----
            threshold_line = alt.Chart(
                pd.DataFrame({"y": [risk_threshold]})
            ).mark_rule(color="red", strokeDash=[5, 5]).encode(y="y:Q")

            risk_chart = (
                alt.Chart(df_last)
                .mark_line(point=False)
                .encode(
                    x=alt.X("time:Q", title="Time step"),
                    y=alt.Y(
                        "failure_prob:Q",
                        scale=alt.Scale(domain=[0, 1]),
                        title="Failure probability",
                    ),
                    color=alt.condition(
                        alt.datum.failure_prob >= risk_threshold,
                        alt.value("red"),
                        alt.value("steelblue"),
                    ),
                )
                .properties(height=250, width="container")
            )
            risk_chart_placeholder.altair_chart(
                risk_chart + threshold_line, use_container_width=True
            )

            # ---- 5. Sensors chart ----
            sensors_df = df_last.melt(
                id_vars=["time"],
                value_vars=[
                    "Torque_(Nm)",
                    "Air_temperature_(K)",
                    "Process_temperature_(K)",
                ],
                var_name="sensor",
                value_name="value",
            )
            sensors_chart = (
                alt.Chart(sensors_df)
                .mark_line(point=False)
                .encode(
                    x=alt.X("time:Q", title="Time step"),
                    y=alt.Y("value:Q", title="Sensor value"),
                    color=alt.Color("sensor:N", title="Sensor"),
                )
                .properties(height=250, width="container")
            )
            sensors_chart_placeholder.altair_chart(
                sensors_chart, use_container_width=True
            )

            # ---- 6. Status + alerts ----
            if prob_live >= critical_threshold:
                status_placeholder.error(
                    f"🚨 CRITICAL RISK: {prob_live*100:.1f}%  (step {len(df_plot)})"
                )
            else:
                current_label = "HIGH RISK ⚠️" if prob_live >= risk_threshold else "OK ✅"
                status_placeholder.markdown(
                    f"**Step {len(df_plot)}** &nbsp;&nbsp; "
                    f"Failure probability: **{prob_live*100:.1f}%** &nbsp;&nbsp; "
                    f"Status: **{current_label}**"
                )

            # ---- 7. Event logging: high risk + sudden spikes ----
            # High risk event
            if prob_live >= risk_threshold:
                st.session_state["events"].append(
                    {
                        "time": int(df_plot.index[-1]),
                        "failure_prob": prob_live,
                        "type": "High risk",
                    }
                )

            # Sudden spike detection
            if len(st.session_state["live_data"]) > 5:
                recent_probs = (
                    st.session_state["live_data"]["failure_prob"].tail(5).values
                )
                prob_change = prob_live - np.mean(recent_probs)
                if prob_change > 0.2:
                    st.session_state["events"].append(
                        {
                            "time": int(df_plot.index[-1]),
                            "failure_prob": prob_live,
                            "type": "Sudden spike ⚡",
                        }
                    )

            if st.session_state["events"]:
                events_df = pd.DataFrame(st.session_state["events"])
                events_df["Risk %"] = (events_df["failure_prob"] * 100).round(1)
                events_df = events_df[["time", "Risk %", "type"]].tail(10)
                events_placeholder.dataframe(
                    events_df,
                    use_container_width=True,
                    height=220,
                )
            else:
                events_placeholder.info(
                    "No high-risk or anomaly events detected yet in this session."
                )

            # ---- 8. Summary metrics ----
            if len(df_plot) > 0:
                high_risk_events = [
                    e for e in st.session_state["events"] if e.get("type") == "High risk"
                ]
                with summary_placeholder:
                    col_stat1, col_stat2 = st.columns(2)
                    with col_stat1:
                        st.metric(
                            "Avg Risk",
                            f"{df_plot['failure_prob'].mean()*100:.1f}%",
                        )
                        st.metric(
                            "Max Risk",
                            f"{df_plot['failure_prob'].max()*100:.1f}%",
                        )
                    with col_stat2:
                        st.metric("Total Steps", len(df_plot))
                        st.metric("High-Risk Events", len(high_risk_events))

            # ---- 9. Progress bar + delay ----
            progress_bar.progress((step_idx + 1) / float(n_steps))
            time.sleep(float(delay))

        # After loop, mark simulation as stopped so user can start again
        st.session_state["sim_running"] = False
        progress_bar.empty()
