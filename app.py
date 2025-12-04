import streamlit as st
import numpy as np
import joblib
from openai import OpenAI
import os
import pandas as pd
from dotenv import load_dotenv
import time
import altair as alt

# Import our custom modules
from styles import get_custom_css
from result_boxes import create_result_box, create_metric_cards, create_status_badge, create_report_box

load_dotenv()

 
# PAGE CONFIG
 
st.set_page_config(
    page_title="Predictive Maintenance Assistant",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# APPLY CUSTOM STYLES
st.markdown(get_custom_css(), unsafe_allow_html=True)

 
# LOAD MODEL
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

xgb = joblib.load(os.path.join(BASE_DIR, "xgb_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
feature_names = joblib.load(os.path.join(BASE_DIR, "feature_names.pkl"))


# GROQ CLIENT
 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GROQ_API_KEY:
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )

def groq_maintenance_report(features_dict, prediction, prob):
    """Generate AI maintenance report"""
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

 
 
st.markdown('<h1 class="main-title">🔧 Predictive Maintenance Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered machine health monitoring and failure prediction</p>', unsafe_allow_html=True)


# TABS
 
tab1, tab2, tab3 = st.tabs(
    ["📈 Failure Risk Calculator", "💬 Maintenance Chatbot", "📡 Live Monitoring"]
)

 
# TAB 1: Risk Calculator
with tab1:
    st.markdown('<div class="section-header">📊 Machine Health Assessment</div>', unsafe_allow_html=True)
    
    st.info("💡 **Tip:** Adjust the sensor readings below to analyze different machine conditions")
    
    # Input section
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🌡️ Temperature Sensors**")
        air_temp = st.number_input(
            "Air Temperature (K)",
            min_value=250.0,
            max_value=350.0,
            value=300.0
        )
        process_temp = st.number_input(
            "Process Temperature (K)",
            min_value=250.0,
            max_value=400.0,
            value=310.0
        )

    with col2:
        st.markdown("**⚙️ Operational Parameters**")
        rotational_speed = st.number_input(
            "Rotational Speed (RPM)",
            min_value=500.0,
            max_value=3000.0,
            value=1500.0
        )
        torque = st.number_input(
            "Torque (Nm)",
            min_value=0.0,
            max_value=100.0,
            value=40.0
        )

    with col3:
        st.markdown("**🔧 Tool Conditions**")
        tool_wear = st.number_input(
            "Tool Wear (minutes)",
            min_value=0.0,
            max_value=300.0,
            value=100.0
        )
        type_l = st.selectbox("Type L", options=[0, 1], index=0)
        type_m = st.selectbox("Type M", options=[0, 1], index=0)

    # Derived features
    temp_delta = process_temp - air_temp
    power_est = torque * rotational_speed

    # Build feature vector
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

    x_vec = [float(input_dict.get(fname, 0.0)) for fname in feature_names]
    x_vec = np.array(x_vec).reshape(1, -1)

    st.markdown("---")
    
    # Beautiful predict button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_btn = st.button("🔍 Analyze Machine Health", use_container_width=True)

    if predict_btn:
        # Make prediction
        with st.spinner("⚙️ Analyzing machine data..."):
            prob = float(xgb.predict_proba(x_vec)[0, 1])
            pred = int(prob >= 0.5)
        
        st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)
        
        # Beautiful result boxes
        card1, card2, card3 = create_metric_cards(prob, pred)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(card1, unsafe_allow_html=True)
        with col2:
            st.markdown(card2, unsafe_allow_html=True)
        with col3:
            st.markdown(card3, unsafe_allow_html=True)
        
        # Additional metrics in beautiful boxes
        st.markdown("---")
        st.markdown("**📊 Derived Metrics**")
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            temp_box = create_result_box(
                f"{temp_delta:.2f} K",
                "Temperature Delta",
                box_type="info"
            )
            st.markdown(temp_box, unsafe_allow_html=True)
        
        with col_d2:
            power_box = create_result_box(
                f"{power_est:.0f} W",
                "Estimated Power",
                box_type="info"
            )
            st.markdown(power_box, unsafe_allow_html=True)
        
        # AI Report in beautiful box
        if GROQ_API_KEY:
            st.markdown('<div class="section-header">🤖 AI Maintenance Recommendations</div>', unsafe_allow_html=True)
            with st.spinner("🤖 Generating personalized report..."):
                report = groq_maintenance_report(input_dict, pred, prob)
            
            report_box = create_report_box(report)
            st.markdown(report_box, unsafe_allow_html=True)
        else:
            st.warning("⚙️ Set GROQ_API_KEY to enable AI recommendations.")


# TAB 2: Chatbot
 
with tab2:
    st.markdown('<div class="section-header">💬 Ask the Maintenance Expert</div>', unsafe_allow_html=True)
    
    st.info(
        "🤖 **Expert AI Assistant** — Ask questions about predictions, features, "
        "or maintenance strategies."
    )
    
    with st.expander("💡 Example Questions"):
        st.markdown("""
        - Why is high torque dangerous?
        - How can we reduce false alarms?
        - What features matter most?
        - What maintenance schedule do you recommend?
        """)

    user_question = st.text_area(
        "Your question:",
        placeholder="Type your maintenance question here...",
        height=130
    )

    col_ask1, col_ask2, col_ask3 = st.columns([1, 2, 1])
    with col_ask2:
        ask_btn = st.button("🚀 Ask RiskBot", use_container_width=True)

    if ask_btn:
        if not GROQ_API_KEY:
            st.error("⚠️ GROQ_API_KEY not configured.")
        elif not user_question.strip():
            st.warning("📝 Please enter a question first.")
        else:
            context = f"""
            We built a predictive maintenance model using features:
            {feature_names}

            XGBoost model with ~82% recall and ~99% accuracy.
            Key features: torque, speed, tool wear, temperature delta, power.
            """
            prompt = f"""
            You are a predictive maintenance expert.

            Context: {context}

            User question: {user_question}

            Answer in 2–4 short paragraphs, simple language, practical advice.
            """

            with st.spinner("🤔 RiskBot is thinking..."):
                response = client.responses.create(
                    model="openai/gpt-oss-20b",
                    input=prompt,
                )
                answer = response.output_text

            st.markdown('<div class="section-header">💡 Expert Response</div>', unsafe_allow_html=True)
            report_box = create_report_box(answer)
            st.markdown(report_box, unsafe_allow_html=True)

 
# TAB 3: Live Monitoring
 

with tab3:
    st.markdown('<div class="section-header">📡 Real-Time Risk Monitoring</div>', unsafe_allow_html=True)
    st.info(
        "🔄 This simulates live sensor data streaming. The XGBoost model evaluates machine health "
        "at each step, similar to an online predictive maintenance dashboard."
    )

 
    if "live_data" not in st.session_state:
        st.session_state["live_data"] = pd.DataFrame(
            columns=[
                "time", "Air_temperature_(K)", "Process_temperature_(K)",
                "Rotational_speed_(rpm)", "Torque_(Nm)", "Tool_wear_(min)",
                "Temp_delta", "Power_est", "failure_prob",
            ]
        )
    if "events" not in st.session_state:
        st.session_state["events"] = []  
    if "sim_load" not in st.session_state:
        st.session_state["sim_load"] = 0.3
    if "sim_wear" not in st.session_state:
        st.session_state["sim_wear"] = 50.0
    if "sim_running" not in st.session_state:
        st.session_state["sim_running"] = False

    risk_threshold = 0.6
    critical_threshold = 0.8

    #  CONFIG CONTROLS 
    st.markdown("**⚙️ Simulation Configuration**")
    scenario = st.selectbox(
        "Load Scenario",
        ["Normal operation", "Increasing load", "High stress", "Random fluctuation"],
        help="Preset patterns for simulated machine load behavior"
    )

    col_controls = st.columns(3)
    with col_controls[0]:
        n_steps = st.number_input(
            "Steps to simulate",
            min_value=20,
            max_value=300,
            value=80,
            step=20,
            help="Number of time steps in this run"
        )
    with col_controls[1]:
        delay = st.number_input(
            "Update delay (sec)",
            min_value=0.1,
            max_value=3.0,
            value=0.4,
            step=0.1,
            help="Time interval between updates"
        )
    with col_controls[2]:
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state["live_data"] = st.session_state["live_data"].iloc[0:0]
            st.session_state["events"] = []
            st.session_state["sim_load"] = 0.3
            st.session_state["sim_wear"] = 50.0
            st.rerun()

    def start_sim():
        st.session_state["sim_running"] = True

    def stop_sim():
        st.session_state["sim_running"] = False

    # START / STOP  
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        start_btn = st.button(

            "▶️ Start Monitoring",
            on_click=start_sim,
            disabled=st.session_state["sim_running"],
            use_container_width=True,
            key='start_btn'
        )
    with col_btn2:
        stop_btn = st.button(
            "⏸️ Stop Monitoring",
            on_click=stop_sim,
            disabled=not st.session_state["sim_running"],
            use_container_width=True,
             key="stop_btn",
        )

    if start_btn:
        st.session_state["sim_running"] = True
    if stop_btn:
        st.session_state["sim_running"] = False
        st.rerun()

    st.markdown("---")

    #  LAYOUT: CHARTS LEFT, STATUS RIGHT  
    left_col, right_col = st.columns([2.5, 1])

    with left_col:
        st.markdown("**📈 Failure Risk Over Time**")
        with st.container():
            risk_chart_placeholder = st.empty()

        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

        st.markdown("**📊 Sensor Trends**")
        with st.container():
            sensors_chart_placeholder = st.empty()

    with right_col:
        st.markdown("**📌 Current Status**")
        status_placeholder = st.empty()

        st.markdown("**📊 Session Summary**")
        summary_placeholder = st.empty()

        st.markdown("**⚠️ Event Log (last 10)**")
        events_placeholder = st.empty()

        if len(st.session_state["live_data"]) > 0:
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            csv = st.session_state["live_data"].to_csv(index=False)
            st.download_button(
                "📥 Download Monitoring Data",
                csv,
                "live_monitoring.csv",
                "text/csv",
                use_container_width=True
            )

    # Progress bar at bottom
    progress_bar = st.progress(0.0)

    #  SIMULATION LOOP 
    if st.session_state["sim_running"]:
        for step_idx in range(int(n_steps)):
            
            if not st.session_state["sim_running"]:
                break

            
            load = st.session_state["sim_load"]

            if scenario == "Normal operation":
                load += np.random.normal(loc=0.0, scale=0.03)
            elif scenario == "Increasing load":
                load += 0.01 + np.random.normal(0.0, 0.02)
            elif scenario == "Under High stress":
                load = np.random.uniform(0.7, 1.0)
            else:  
                load = np.random.uniform(0.0, 1.0)

            load = float(np.clip(load, 0.0, 1.0))
            st.session_state["sim_load"] = load

            wear = st.session_state["sim_wear"] + np.random.uniform(0.2, 0.8)
            wear = float(np.clip(wear, 0.0, 300.0))
            st.session_state["sim_wear"] = wear

            # Base operating points
            base_air = 295.0
            base_speed = 1200.0
            base_torque = 30.0

            # Simulated sensors (correlated with load)
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

            #   2. Model prediction  
            x_vec_live = [float(feat_dict.get(fname, 0.0)) for fname in feature_names]
            x_vec_live = np.array(x_vec_live).reshape(1, -1)
            prob_live = float(xgb.predict_proba(x_vec_live)[0, 1])

            #   3. Append row to live_data  
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
            df_last = df_plot.tail(100)

             
            df_last_reset = df_last.reset_index()

            #  Failure probability chart (top, big)  
            threshold_df = pd.DataFrame({"y": [risk_threshold]})

            risk_line = (
                alt.Chart(df_last_reset)
                .mark_line(point=False)
                .encode(
                    x=alt.X("time:Q", title="Time step"),
                    y=alt.Y(
                        "failure_prob:Q",
                        title="Failure probability",
                        scale=alt.Scale(domain=[0, 1]),
                    ),
                )
                .properties(
                    height=280,          # big main chart
                    width="container",
                              # we use Streamlit heading above
                )
            )

            threshold_line = (
                alt.Chart(threshold_df)
                .mark_rule(color="red", strokeDash=[6, 4])
                .encode(y="y:Q")
            )

            risk_chart_placeholder.altair_chart(
                risk_line + threshold_line,
                use_container_width=True,
            )

       
            sensors_df = df_last_reset.melt(
                id_vars="time",
                value_vars=[
                    "Air_temperature_(K)",
                    "Process_temperature_(K)",
                    "Torque_(Nm)",
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
                .properties(
                    height=230,          # shorter chart
                    width="container",
                )
            )

            sensors_chart_placeholder.altair_chart(
                sensors_chart,
                use_container_width=True,
            )


            
            #   5. Current status with simple badges  
            if prob_live >= critical_threshold:
                status_placeholder.error(
                    f"🚨 CRITICAL RISK: {prob_live*100:.1f}% (step {len(df_plot)})"
                )
            elif prob_live >= risk_threshold:
                status_placeholder.warning(
                    f"⚠️ HIGH RISK: {prob_live*100:.1f}% (step {len(df_plot)})"
                )
            else:
                status_placeholder.success(
                    f"✅ OK: Failure probability {prob_live*100:.1f}% (step {len(df_plot)})"
                )

            #   6. Event logging (high risk + spikes)  
            if prob_live >= risk_threshold:
                st.session_state["events"].append(
                    {
                        "time": int(df_plot.index[-1]),
                        "failure_prob": prob_live,
                        "type": "High risk",
                    }
                )

            if len(st.session_state["live_data"]) > 5:
                recent_probs = st.session_state["live_data"]["failure_prob"].tail(5).values
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
                events_placeholder.info("No high-risk or anomaly events detected yet.")

            #  7. Session summary metrics  
            if len(df_plot) > 0:
                high_risk_events = [
                    e for e in st.session_state["events"] if e.get("type") == "High risk"
                ]
                with summary_placeholder:
                    col_stat1, col_stat2 = st.columns(2)
                    with col_stat1:
                        st.metric("Avg Risk", f"{df_plot['failure_prob'].mean()*100:.1f}%")
                        st.metric("Max Risk", f"{df_plot['failure_prob'].max()*100:.1f}%")
                    with col_stat2:
                        st.metric("Total Steps", len(df_plot))
                        st.metric("High-Risk Events", len(high_risk_events))

            # 8. Progress bar + delay  
            progress_bar.progress((step_idx + 1) / float(n_steps))
            time.sleep(float(delay))

        # After loop, stop monitoring so user can start again
        st.session_state["sim_running"] = False
        progress_bar.empty()