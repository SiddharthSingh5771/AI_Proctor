import streamlit as st
import cv2
import time
import os

IS_CLOUD = os.environ.get("STREAMLIT_SERVER_RUNNING") == "1"

import pandas as pd

# Import your existing proctor logic
from ai_procter import AIProctorFull, AudioMonitor

# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="AI Proctoring System",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 AI-Based Online Exam Proctoring System")
st.markdown(
    """
    This system monitors students during online examinations using  
    **Computer Vision and Audio Analysis** to detect cheating behavior.
    """
)

# ----------------------------------
# Sidebar Controls
# ----------------------------------
st.sidebar.header("🎛 Controls")

start_button = st.sidebar.button("▶ Start Proctoring")
stop_button = st.sidebar.button("⏹ Stop Proctoring")

st.sidebar.markdown("---")
st.sidebar.subheader("📌 Live Events")

status_face = st.sidebar.empty()
status_look = st.sidebar.empty()
status_person = st.sidebar.empty()
status_phone = st.sidebar.empty()
status_noise = st.sidebar.empty()

st.sidebar.markdown("---")
risk_display = st.sidebar.empty()

# ----------------------------------
# Layout
# ----------------------------------
col1, col2 = st.columns([2, 1])

video_placeholder = col1.empty()
col2.subheader("📊 Live Status")
info_box = col2.empty()

# ----------------------------------
# Session State Initialization
# ----------------------------------
if "running" not in st.session_state:
    st.session_state.running = False

if "proctor" not in st.session_state:
    st.session_state.proctor = None

if "audio" not in st.session_state:
    st.session_state.audio = None

if "log_file" not in st.session_state:
    st.session_state.log_file = None

# ----------------------------------
# Start Proctoring
# ----------------------------------
if start_button and not st.session_state.running:
    st.session_state.running = True
    st.session_state.proctor = AIProctorFull()
    st.session_state.audio = AudioMonitor()
    st.session_state.audio.start()

    st.session_state.log_file = st.session_state.proctor.log_filename
    st.success("✅ Proctoring started")

# ----------------------------------
# Stop Proctoring
# ----------------------------------
if stop_button and st.session_state.running:
    st.session_state.running = False
    st.session_state.audio.stop()
    st.session_state.proctor.close()
    st.success("🛑 Proctoring stopped. Report generated.")

# ----------------------------------
# Video Processing Loop
# ----------------------------------
if st.session_state.running:

    if IS_CLOUD:
        st.warning("Webcam access is disabled on Streamlit Cloud.")
        st.stop()

    cap = cv2.VideoCapture(0)


    if not cap.isOpened():
        st.error("❌ Unable to access webcam")
        st.session_state.running = False

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to read webcam frame")
            break

        noise_flag = st.session_state.audio.is_noise_detected()
        annotated, events, risk = st.session_state.proctor.update_state(
            frame, noise_flag
        )

        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB")

        # Sidebar updates
        status_face.markdown(f"👤 Face detected: **{not events['no_face']}**")
        status_look.markdown(f"👀 Looking away: **{events['looking_away']}**")
        status_person.markdown(f"👥 Extra person: **{events['extra_person']}**")
        status_phone.markdown(f"📱 Phone detected: **{events['phone_detected']}**")
        status_noise.markdown(f"🔊 Noise detected: **{events['noise_detected']}**")

        risk_display.markdown(f"## 🚨 Risk Score: `{risk:.2f}`")

        info_box.markdown(
            f"""
            **Session Details**
            - Risk Score: `{risk:.2f}`
            - Looking Away: `{events['looking_away']}`
            - Extra Person: `{events['extra_person']}`
            - Phone Detected: `{events['phone_detected']}`
            - Noise Detected: `{events['noise_detected']}`
            """
        )

        time.sleep(0.03)

    cap.release()
    cv2.destroyAllWindows()

else:
    st.info("Click **Start Proctoring** to begin monitoring.")

# ----------------------------------
# REPORT SECTION
# ----------------------------------
st.markdown("---")
st.header("📄 Proctoring Session Report")

if st.session_state.log_file and os.path.exists(st.session_state.log_file):

    df = pd.read_csv(st.session_state.log_file)

    if len(df) > 0:
        total_time = df["session_time_sec"].max()
        final_risk = df["risk_score"].iloc[-1]

        colA, colB = st.columns(2)

        with colA:
            st.metric("⏱ Total Session Time (sec)", f"{total_time:.1f}")
            st.metric("🚨 Final Risk Score", f"{final_risk:.2f}")

        with colB:
            st.metric("👀 Looking Away Events", int(df["looking_away"].sum()))
            st.metric("👥 Extra Person Events", int(df["extra_person"].sum()))
            st.metric("📱 Phone Detected Events", int(df["phone_detected"].sum()))
            st.metric("🔊 Noise Events", int(df["noise_detected"].sum()))

        st.subheader("📊 Log Preview (Last 20 rows)")
        st.dataframe(df.tail(20), use_container_width=True)

        with open(st.session_state.log_file, "rb") as f:
            st.download_button(
                label="⬇ Download Full CSV Report",
                data=f,
                file_name=os.path.basename(st.session_state.log_file),
                mime="text/csv"
            )
    else:
        st.warning("Log file is empty.")
else:
    st.info("No report available yet. Start and stop proctoring to generate a report.")
