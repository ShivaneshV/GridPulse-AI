"""
GridPulse AI: Sovereign Grid Intelligence Platform Dashboard
This Streamlit app provides the Mission Control interface.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import os
import json
import sys
from datetime import datetime, timedelta
from fpdf import FPDF
import io
from streamlit_autorefresh import st_autorefresh

# Ensure src is in path to import mock modules
sys.path.append(os.path.abspath("src"))

# Theme Colors
ELECTRIC_BLUE = "#00E5FF"
SAFETY_ORANGE = "#FF6D00"
DARK_NAVY = "#0B132B"

st.set_page_config(
    page_title="GridPulse AI - Mission Control",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI Polish
st.markdown(f"""
<style>
    .metric-card {{
        background-color: #1C2541;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid {ELECTRIC_BLUE};
        margin-bottom: 20px;
    }}
    .alert-card {{
        background-color: #1C2541;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid {SAFETY_ORANGE};
        margin-bottom: 20px;
    }}
    .dlms-feed {{
        font-family: 'Courier New', monospace;
        font-size: 0.8em;
        color: #00E5FF;
        background-color: #0B132B;
        padding: 10px;
        border-radius: 5px;
        height: 300px;
        overflow-y: auto;
    }}
    .peer-validation {{
        background-color: rgba(0, 229, 255, 0.1);
        border: 1px solid #00E5FF;
        border-radius: 5px;
        padding: 10px;
        font-size: 0.9em;
        margin-top: 10px;
    }}
    /* Ensure Streamlit Header and Deploy button are visible */
    header {{visibility: visible !important;}}
    .stDeployButton {{display: block !important;}}
    [data-testid="stToolbar"] {{visibility: visible !important;}}
    
    /* Hide Streamlit Footer only */
    footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# PDF Report Generator
# ---------------------------------------------------------------------
def generate_forensic_pdf(node_id, lat, lon, anomaly_data):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", style="B", size=16)
    pdf.cell(200, 10, txt="GridPulse AI - Digital Evidence Report", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("Helvetica", size=12)
    pdf.cell(200, 10, txt=f"Meter ID: {node_id}", ln=True)
    ubid_mock = f"KA-SWS-{node_id.split('_')[-1]}492"
    pdf.cell(200, 10, txt=f"Mapped UBID (K-Commerce): {ubid_mock}", ln=True)
    pdf.cell(200, 10, txt=f"GPS Coordinates: {lat:.4f}, {lon:.4f}", ln=True)
    pdf.cell(200, 10, txt=f"Status: CRITICAL - {anomaly_data['verdict'].split(' of ')[-1]}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence Verdict: {anomaly_data['verdict'].split(' ')[0]}", ln=True)
    pdf.cell(200, 10, txt=f"Signature Detected: {anomaly_data['signature']}", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Helvetica", style="B", size=12)
    pdf.cell(200, 10, txt="Peer-Group Validation:", ln=True)
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, txt=anomaly_data['peer_logic'])
    
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Action Required: {anomaly_data['action']}", ln=True)
    
    # Return as bytes
    return bytes(pdf.output())

# ---------------------------------------------------------------------
# Data Loading & Simulation Logic
# ---------------------------------------------------------------------
@st.cache_data
def load_sandbox_data():
    # Resolve data path relative to the root of the project
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(base_dir, "data", "synthetic", "sandbox")
    
    # If data doesn't exist, return empty mocks
    if not os.path.exists(data_dir):
        return None, None, None
        
    try:
        with open(os.path.join(data_dir, "topology.json"), 'r') as f:
            topology = json.load(f)
            
        with open(os.path.join(data_dir, "theft_events.json"), 'r') as f:
            theft_events = json.load(f)
            
        meters = {}
        meters_dir = os.path.join(data_dir, "meters")
        for file in os.listdir(meters_dir):
            if file.endswith('.csv'):
                meter_id = file.replace('.csv', '')
                meters[meter_id] = pd.read_csv(os.path.join(meters_dir, file))
                
        return topology, theft_events, meters
    except Exception as e:
        st.error(f"Error loading sandbox data: {e}")
        return None, None, None

topology, theft_events, meters = load_sandbox_data()

# ---------------------------------------------------------------------
# Sidebar: Zero-Hardware Branding & Live Feed
# ---------------------------------------------------------------------
with st.sidebar:
    # Use relative path for logo to work on Cloud deployment
    logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=80)
    else:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/Electric_plug_symbol.svg/200px-Electric_plug_symbol.svg.png", width=50) 
    st.title("GridPulse AI")
    st.markdown("### Sovereign Grid Intelligence")
    
    st.markdown("---")
    st.markdown("🔒 **AES-256 Encryption Active**")
    st.markdown("🔌 **Virtual DLMS Port Connected**")
    st.caption("No hardware modifications detected.")
    
    if 'anomaly_count' not in st.session_state:
        st.session_state['anomaly_count'] = 0

    if st.button("Trigger 'Bypass Hooking' Event", type="primary"):
        st.session_state['trigger_anomaly'] = True
        st.session_state['anomaly_count'] += 1
    else:
        if 'trigger_anomaly' not in st.session_state:
            st.session_state['trigger_anomaly'] = False

    st.markdown("---")
    st.markdown("#### Live Pulse Feed (DLMS/COSEM)")
    
    # Native scrolling container
    feed_container = st.container(height=300)

# ---------------------------------------------------------------------
# Main Layout: Personas
# ---------------------------------------------------------------------
st.title("⚡ GridPulse AI Central Command")

tabs = st.tabs([
    "📍 Field Engineer (AEE): Live Anomaly Detection",
    "📈 Data Analyst: TFT Multi-Horizon Forecast",
    "💰 Government Official (IAS): Revenue Recovery"
])

# Generate mock map data based on topology
if topology:
    map_data = pd.DataFrame(topology['meters'])
    map_data['status'] = 'Normal'
else:
    # Fallback to dummy data
    map_data = pd.DataFrame({
        'id': ['MT_001', 'MT_002', 'MT_003', 'MT_004'],
        'latitude': [12.9716, 12.9720, 12.9690, 12.9750],
        'longitude': [77.5946, 77.5950, 77.5900, 77.5980],
        'status': ['Normal', 'Normal', 'Normal', 'Normal']
    })

if st.session_state['trigger_anomaly']:
    num_anomalies = min(6, len(map_data))
    for i in range(num_anomalies):
        map_data.loc[map_data.index[i], 'status'] = 'Anomaly'

# --- Tab 1: Field Engineer ---
with tabs[0]:
    st.header("Live Anomaly Detection & Routing")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Grid Node Map")
        # Map visualization
        fig = px.scatter_mapbox(
            map_data, lat="latitude", lon="longitude", hover_name="id",
            color="status", 
            color_discrete_map={"Normal": ELECTRIC_BLUE, "Anomaly": SAFETY_ORANGE},
            size_max=15, zoom=13, height=500
        )
        fig.update_layout(mapbox_style="carto-darkmatter")
        fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0}) # Added top margin to prevent overlap
        # Add pulsing effect for anomaly (simulated by larger marker)
        if st.session_state['trigger_anomaly']:
            fig.update_traces(marker=dict(size=[25 if s == 'Anomaly' else 10 for s in map_data['status']]))
        # Hide the redundant Plotly modebar and explicitly enable scroll zoom
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'scrollZoom': True})
        
    with col2:
        st.subheader("Forensic Alerts")
        if st.session_state['trigger_anomaly']:
            count = st.session_state.get('anomaly_count', 1)
            anomalies = [
                {
                    "verdict": "94% Confidence of Meter Bypass",
                    "signature": "300Hz Harmonic Distortion Spike",
                    "action": f"Inspect Pole B-402",
                    "peer_logic": f"Consumption is 85% below its Peer Group (Same Transformer/Weather Cluster)."
                },
                {
                    "verdict": "89% Confidence of Phase Tampering",
                    "signature": "Asymmetric Voltage Drop on Phase B",
                    "action": f"Dispatch crew to inspect neutral line",
                    "peer_logic": f"Neighboring nodes on the same feeder show perfect phase balance, ruling out upstream issues."
                },
                {
                    "verdict": "97% Confidence of Magnetic Interference",
                    "signature": "Zero recorded current with normal 230V voltage",
                    "action": f"Investigate for Neodymium magnet placement",
                    "peer_logic": f"Historical load curve does not match sudden zero-drop during peak operating hours."
                },
                {
                    "verdict": "91% Confidence of Direct Line Tapping",
                    "signature": "Unmetered load spike detected upstream",
                    "action": f"Inspect overhead lines near Transformer",
                    "peer_logic": f"Transformer total load exceeds sum of metered loads by 15kW, isolating the theft locally."
                },
                {
                    "verdict": "95% Confidence of EV Charger Overload",
                    "signature": "Sustained 22kW square-wave load detected",
                    "action": f"Issue notice for unregistered Tier-2 EV charging",
                    "peer_logic": f"Node exceeds sanctioned 5kW limit by 400% while peer cluster remains baseline."
                },
                {
                    "verdict": "88% Confidence of Illegal Solar Backfeed",
                    "signature": "Reverse power flow with unmatched phase angle",
                    "action": f"Check for unauthorized inverter grid-tie",
                    "peer_logic": f"Node is pushing power into the grid despite not being a registered prosumer."
                },
                {
                    "verdict": "99% Confidence of Firmware Tampering",
                    "signature": "DLMS/COSEM Cryptographic Hash Mismatch",
                    "action": f"Deploy cyber-security team to physically replace Smart Meter",
                    "peer_logic": f"Telemetry packets failed AES-256 decryption validation on the central server."
                },
                {
                    "verdict": "92% Confidence of Loose Neutral Connection",
                    "signature": "Rapid voltage swing (180V to 260V) across phases",
                    "action": f"Emergency maintenance: Tighten neutral clamp to prevent appliance fires",
                    "peer_logic": f"Condition isolated to single node; transformer neutral is stable."
                },
                {
                    "verdict": "86% Confidence of Transformer Oil Degradation",
                    "signature": "High-frequency partial discharge acoustic signature",
                    "action": f"Schedule predictive maintenance for DGA (Dissolved Gas Analysis) at upstream TR-04",
                    "peer_logic": f"Harmonic distortion detected simultaneously across ALL downstream peer nodes."
                },
                {
                    "verdict": "94% Confidence of Tariff Violation & UBID Mismatch",
                    "signature": "Sustained 12kW flat load during 9AM-5PM",
                    "action": f"Flag UBID for 'Dormant to Active' status change",
                    "peer_logic": f"K-Commerce UBID registry lists this business as 'Closed/Dormant', but load shape matches active industrial manufacturing."
                }
            ]
            
            num_anomalies = min(6, len(map_data))
            for i in range(num_anomalies):
                node_id = map_data.iloc[i]['id']
                lat = map_data.iloc[i]['latitude']
                lon = map_data.iloc[i]['longitude']
                
                anomaly = anomalies[(count + i) % len(anomalies)]
                
                with st.expander(f"🚨 {node_id}: {anomaly['verdict'].split(' of ')[-1]}", expanded=(i==0)):
                    ubid_mock = f"KA-SWS-{node_id.split('_')[-1]}492"
                    st.markdown(f"""
                    <div style="margin-top:0;">
                        <p><strong>Mapped UBID (K-Commerce):</strong> <span style="color:#00E5FF;">{ubid_mock}</span></p>
                        <p><strong>Verdict:</strong> {anomaly['verdict']}</p>
                        <p><strong>Signature:</strong> {anomaly['signature']}</p>
                        <p><strong>Action Required:</strong> {anomaly['action']} (GPS: {lat:.4f}, {lon:.4f})</p>
                        
                        <div class="peer-validation">
                            <strong>✔️ Peer-Group Validation:</strong><br>
                            {anomaly['peer_logic']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # PDF Download Button
                    pdf_bytes = generate_forensic_pdf(node_id, lat, lon, anomaly)
                    st.download_button(
                        label=f"📄 Download Report",
                        data=pdf_bytes,
                        file_name=f"Forensic_Report_{node_id}.pdf",
                        mime="application/pdf",
                        type="primary",
                        key=f"dl_btn_{node_id}"
                    )
            
            # Spectral FFT Mock Chart
            st.markdown("#### Spectral FFT Analysis")
            x = np.linspace(0, 500, 500)
            y = np.exp(-(x-50)**2 / 10) + 0.1*np.random.rand(500)
            # Inject 300Hz anomaly
            y += 0.8 * np.exp(-(x-300)**2 / 5)
            
            fig_fft = go.Figure()
            fig_fft.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color=SAFETY_ORANGE), fill='tozeroy'))
            fig_fft.update_layout(
                paper_bgcolor=DARK_NAVY, plot_bgcolor=DARK_NAVY,
                font=dict(color="white"), height=200, margin={"r":0,"t":0,"l":0,"b":0},
                xaxis_title="Frequency (Hz)", yaxis_title="Magnitude"
            )
            st.plotly_chart(fig_fft, use_container_width=True)
            
        else:
            st.markdown("""
            <div class="metric-card">
                <h3 style="color:#00E5FF; margin-top:0;">System Normal</h3>
                <p>All monitored nodes operating within expected spectral parameters.</p>
            </div>
            """, unsafe_allow_html=True)

# --- Tab 2: Data Analyst ---
with tabs[1]:
    st.header("Temporal Fusion Transformer (TFT) Insights")
    st.markdown("Compare multi-horizon AI forecasts against actual synthetic grid consumption. *(Live Streaming)*")
    
    # Auto-refresh the page every 1 second (1000 milliseconds)
    refresh_count = st_autorefresh(interval=1000, limit=None, key="tft_autorefresh")
    
    # Mock TFT Forecast Data - Animate the sine wave based on the refresh count
    offset = refresh_count * 0.2  # Shift the wave forward every second
    dates = pd.date_range(end=datetime.now(), periods=48, freq='s') # Updates every second
    
    actual = 50 + 20 * np.sin(np.linspace(0 + offset, 4*np.pi + offset, 48)) + np.random.normal(0, 2, 48)
    forecast = 50 + 20 * np.sin(np.linspace(0 + offset, 4*np.pi + offset, 48))
    
    fig_tft = go.Figure()
    fig_tft.add_trace(go.Scatter(x=dates, y=actual, name="Actual Load (kW)", line=dict(color=ELECTRIC_BLUE)))
    fig_tft.add_trace(go.Scatter(x=dates[-24:], y=forecast[-24:], name="TFT Forecast (kW)", line=dict(color=SAFETY_ORANGE, dash='dash')))
    fig_tft.add_trace(go.Scatter(
        x=dates[-24:].tolist() + dates[-24:][::-1].tolist(),
        y=(forecast[-24:] + 5).tolist() + (forecast[-24:] - 5)[::-1].tolist(),
        fill='toself', fillcolor='rgba(255,109,0,0.2)', line=dict(color='rgba(255,255,255,0)'),
        showlegend=False, name='95% Confidence Interval'
    ))
    
    # Keep the Y-axis fixed so the wave looks like it's flowing through a window
    fig_tft.update_layout(
        paper_bgcolor=DARK_NAVY, plot_bgcolor=DARK_NAVY, font=dict(color="white"),
        height=400, yaxis=dict(range=[20, 80])
    )
    st.plotly_chart(fig_tft, use_container_width=True)
    
    st.markdown("#### TFT Attention Heads (Feature Importance)")
    features = ['Historical Load', 'Hour of Day', 'Day of Week', 'Temperature', 'Voltage']
    importance = [0.45, 0.25, 0.15, 0.10, 0.05]
    fig_shap = px.bar(x=importance, y=features, orientation='h', color_discrete_sequence=[ELECTRIC_BLUE])
    fig_shap.update_layout(
        paper_bgcolor=DARK_NAVY, plot_bgcolor=DARK_NAVY, font=dict(color="white"), height=300
    )
    st.plotly_chart(fig_shap, use_container_width=True)

# --- Tab 3: Government Official ---
with tabs[2]:
    st.header("Revenue Recovery Dashboard")
    st.markdown("Quantifying the financial impact of GridPulse AI across the simulated cluster.")
    
    col_a, col_b, col_c = st.columns(3)
    
    if st.session_state['trigger_anomaly']:
        count = max(1, st.session_state['anomaly_count'])
        recovered = f"₹ {14.5 + (count - 1) * 1.2:.1f} Lakhs"
        events = f"{12 + (count - 1)}"
        loss_reduction = f"{4.2 + (count - 1) * 0.1:.1f}%"
    else:
        recovered = "₹ 0"
        events = "0"
        loss_reduction = "0%"

    with col_a:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{recovered}</h2>
            <p>Potential Revenue Recovered (Last 30 Days)</p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{loss_reduction}</h2>
            <p>AT&C Loss Reduction (Simulated)</p>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{events}</h2>
            <p>Theft Events Flagged</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### Cumulative Savings Projection")
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    
    # Dramatically warp the curve upwards based on anomaly count to make it visually obvious
    count = st.session_state.get('anomaly_count', 0)
    base_savings = [2, 5.5, 9, 14.5, 21, 28]
    growth_factors = [0, 0.5, 1.2, 2.5, 4.0, 6.0]
    
    savings = [base + (factor * count) for base, factor in zip(base_savings, growth_factors)]
    
    fig_rev = px.area(x=months, y=savings, labels={'x': 'Month', 'y': 'Savings (Lakhs INR)'})
    fig_rev.update_traces(line_color=ELECTRIC_BLUE, fillcolor='rgba(0, 229, 255, 0.3)')
    
    # Fix the Y-axis range to 100 so the graph visually grows taller on screen
    fig_rev.update_layout(
        paper_bgcolor=DARK_NAVY, plot_bgcolor=DARK_NAVY, font=dict(color="white"),
        yaxis=dict(range=[0, 100])
    )
    st.plotly_chart(fig_rev, use_container_width=True)

# ---------------------------------------------------------------------
# Simulate DLMS/COSEM Feed in Sidebar
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Simulate DLMS/COSEM Feed in Sidebar
# ---------------------------------------------------------------------
feed_text = ""
meters_list = ["MT_001", "MT_002", "MT_003", "MT_004", "MT_005", "MT_006"]
# Generate 50 lines so there is a long history to scroll through
for i in range(50):
    m = meters_list[i % len(meters_list)]
    ts = (datetime.now() - timedelta(seconds=i*2)).strftime("%H:%M:%S")
    val = 230 + np.random.normal(0, 2)
    if st.session_state['trigger_anomaly'] and i < 3:
        feed_text = f"[{ts}] {m} OBIS:1.7.0 (THD) - WARNING: LIMIT EXCEEDED\n" + feed_text
    else:
        feed_text = f"[{ts}] {m} OBIS:32.7.0 (Voltage) - {val:.1f}V OK\n" + feed_text
        
# Reverse the generated lines so the newest (i=0) is at the top
lines = feed_text.strip().split("\n")
lines.reverse()
feed_text = "\n".join(lines)

feed_container.code(feed_text, language="log")
