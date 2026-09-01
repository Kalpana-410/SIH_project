import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tempfile
import shutil
import subprocess
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px

# ============================================================
# SMARTTRANSIT AI
# AI-powered public transport crowd monitoring and prediction
# ============================================================

st.set_page_config(
    page_title="SmartTransit AI",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# File paths
# -----------------------------
DATA_PATH = "SmartTransit_AI_Updated_Dataset.csv"

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
/* ============================================================
   GLOBAL SMARTTRANSIT AI THEME
   Applies consistently to every page
   ============================================================ */

/* Full application background */
.stApp {
    background: linear-gradient(135deg, #dbeafe 0%, #eff6ff 45%, #e0f2fe 100%);
    min-height: 100vh;
}

/* Main content background on every page */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #dbeafe 0%, #f0f7ff 50%, #e0f2fe 100%);
}

[data-testid="stAppViewContainer"] > .main {
    background: transparent;
}

[data-testid="stMainBlockContainer"] {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Keep the header area transparent so the page background is visible */
[data-testid="stHeader"] {
    background: transparent;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2f63 0%, #164e9c 55%, #1d4ed8 100%);
}

[data-testid="stSidebar"] * {
    color: #f8fbff !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #dbeafe !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: rgba(255, 255, 255, 0.07);
    border-radius: 9px;
    padding: 7px 10px;
    margin: 3px 0;
    transition: all 0.2s ease;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(255, 255, 255, 0.16);
}

/* Blue active navigation item */
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
}

/* Titles */
.main-title {
    font-size: 36px;
    font-weight: 800;
    margin-bottom: 0;
    color: #0b2a5b;
    letter-spacing: -0.5px;
}

.subtitle {
    color: #35577f;
    font-size: 16px;
    margin-top: 4px;
}

h1, h2, h3 {
    color: #0b2a5b;
}

/* Soft white cards so content remains easy to read over the blue background */
[data-testid="stMetric"],
[data-testid="stDataFrame"],
[data-testid="stFileUploader"],
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.94);
    border: 1px solid #bfdbfe;
    border-radius: 14px;
    box-shadow: 0 5px 18px rgba(30, 64, 175, 0.10);
}

[data-testid="stMetric"] {
    padding: 16px;
}

[data-testid="stMetricLabel"] {
    color: #35577f !important;
}

[data-testid="stMetricValue"] {
    color: #0b2a5b !important;
}

/* Inputs and selectors */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-testid="stNumberInput"] > div {
    border-radius: 10px;
    background: #ffffff;
}

[data-baseweb="select"] > div {
    border-color: #93c5fd;
}

/* All Streamlit buttons: blue */
.stButton > button {
    width: 100%;
    min-height: 44px;
    border-radius: 10px;
    border: 1px solid #2563eb;
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: #ffffff !important;
    font-weight: 700;
    box-shadow: 0 4px 10px rgba(37, 99, 235, 0.22);
    transition: all 0.2s ease;
}

.stButton > button:hover,
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af);
    border-color: #1e40af;
    transform: translateY(-1px);
    box-shadow: 0 6px 14px rgba(37, 99, 235, 0.32);
}

.stButton > button:active {
    transform: translateY(0);
}

/* Primary buttons are also blue (not red) */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    border-color: #2563eb;
    box-shadow: 0 4px 10px rgba(37, 99, 235, 0.22);
}

/* Informational / alert boxes */
[data-testid="stAlert"] {
    border-radius: 12px;
    box-shadow: 0 3px 12px rgba(30, 64, 175, 0.06);
}

/* Data tables */
[data-testid="stDataFrame"] {
    overflow: hidden;
}

/* Progress bars */
[data-testid="stProgressBar"] > div {
    border-radius: 999px;
    background: #dbeafe;
}

[data-testid="stProgressBar"] > div > div {
    border-radius: 999px;
    background: #2563eb;
}

/* Uploaded file area */
[data-testid="stFileUploader"] {
    padding: 6px;
}

/* Code blocks */
[data-testid="stCode"] {
    border-radius: 12px;
}

/* Dividers */
hr {
    border-color: #bfdbfe;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING
# ============================================================


@st.cache_data
def load_data(path):
    df = pd.read_csv(path)

    # Convert date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Make sure important numeric columns are numeric
    numeric_cols = [
        "capacity", "passengers", "occupancy_rate",
        "distance_km", "fare_per_passenger", "revenue",
        "fuel_consumed_liters", "hour", "peak_hour", "weekend",
        "holiday", "event_flag", "next_bus_arrival_min",
        "bus_frequency_min", "current_onboard", "waiting_crowd",
        "boarding_count", "alighting_count",
        "vision_waiting_count", "vision_boarding_count",
        "passengers_after_stop", "occupancy_after_stop",
        "future_passengers", "future_occupancy_rate",
        "waiting_pressure_pct", "available_seats_before_boarding"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"Dataset not found: {DATA_PATH}\n\n"
        "Place SmartTransit_AI_Updated_Dataset.csv in the same folder as app.py."
    )
    st.stop()


# ============================================================
# ML MODEL
# ============================================================

FEATURES = [
    "capacity",
    "current_onboard",
    "waiting_crowd",
    "boarding_count",
    "alighting_count",
    "distance_km",
    "hour",
    "peak_hour",
    "weekend",
    "holiday",
    "event_flag",
    "next_bus_arrival_min",
    "bus_frequency_min",
    "occupancy_after_stop",
    "waiting_pressure_pct"
]

TARGET = "future_passengers"


@st.cache_resource
def train_model(data):
    model_df = data[FEATURES + [TARGET]].dropna().copy()

    X = model_df[FEATURES]
    y = model_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return model, mae, r2


model, model_mae, model_r2 = train_model(df)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def risk_level(occupancy):
    if occupancy < 60:
        return "LOW"
    elif occupancy < 80:
        return "MODERATE"
    elif occupancy <= 100:
        return "HIGH"
    return "OVERCRITICAL"


def risk_icon(level):
    return {
        "LOW": "🟢",
        "MODERATE": "🟡",
        "HIGH": "🟠",
        "OVERCRITICAL": "🔴"
    }[level]


def predict_future(
    current_onboard,
    waiting_crowd,
    boarding,
    alighting,
    capacity,
    distance,
    hour,
    weekend,
    holiday,
    event_flag,
    next_bus_arrival,
    bus_frequency
):
    occupancy_after = (
        max(0, current_onboard - alighting + boarding)
        / capacity * 100
    )

    waiting_pressure = waiting_crowd / capacity * 100

    row = pd.DataFrame([{
        "capacity": capacity,
        "current_onboard": current_onboard,
        "waiting_crowd": waiting_crowd,
        "boarding_count": boarding,
        "alighting_count": alighting,
        "distance_km": distance,
        "hour": hour,
        "peak_hour": int(hour in [7, 8, 9, 17, 18, 19]),
        "weekend": int(weekend),
        "holiday": int(holiday),
        "event_flag": int(event_flag),
        "next_bus_arrival_min": next_bus_arrival,
        "bus_frequency_min": bus_frequency,
        "occupancy_after_stop": occupancy_after,
        "waiting_pressure_pct": waiting_pressure
    }])

    prediction = model.predict(row)[0]
    prediction = max(0, min(prediction, capacity))

    return int(round(prediction))


def _encode_browser_friendly(src_path, dst_path):
    """
    Re-encode an OpenCV-written video to H.264 / yuv420p MP4 so that
    browsers (and therefore st.video) can actually play it.

    Falls back to the original file if ffmpeg is unavailable.
    """
    ffmpeg = shutil.which("ffmpeg")

    if not ffmpeg:
        return src_path

    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(src_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "23",
        "-movflags", "+faststart",
        "-an",
        str(dst_path),
    ]

    try:
        subprocess.run(cmd, check=True)
    except Exception:
        return src_path

    if Path(dst_path).exists() and Path(dst_path).stat().st_size > 0:
        return dst_path

    return src_path


def process_video(video_bytes, model_path="yolo11n.pt",
                  confidence=0.35, line_ratio=0.84,
                  progress_callback=None):
    """
    Actual YOLO + ByteTrack video processing.

    line_ratio is the horizontal position of the bus-entry line
    as a fraction of video width.

    Returns:
        output_video_path, entered_ids, exited_ids, max_waiting_count
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "Ultralytics is not installed. Run: pip install ultralytics"
        )

    temp_dir = Path(tempfile.mkdtemp())
    input_path = temp_dir / "input.mp4"
    raw_path = temp_dir / "raw.mp4"
    output_path = temp_dir / "processed.mp4"

    input_path.write_bytes(video_bytes)

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError("Could not open uploaded video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if fps <= 0 or fps > 120:
        fps = 25

    if width <= 0 or height <= 0:
        raise RuntimeError("Could not read video dimensions.")

    line_x = int(width * line_ratio)

    writer = None

    for codec in ("avc1", "mp4v"):
        candidate = cv2.VideoWriter(
            str(raw_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height)
        )
        if candidate.isOpened():
            writer = candidate
            break
        candidate.release()

    if writer is None:
        raise RuntimeError("Could not create the output video writer.")

    yolo = YOLO(model_path)

    previous_positions = {}
    entered_ids = set()
    exited_ids = set()

    max_waiting_count = 0
    frame_index = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_index += 1

        results = yolo.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=confidence,
            verbose=False
        )

        # Bus-entry line
        cv2.line(frame, (line_x, 0), (line_x, height), (0, 0, 255), 3)

        cv2.putText(
            frame,
            "BUS ENTRY",
            (max(10, line_x - 110), 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        waiting_now = 0

        result = results[0]

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()

            for box, track_id, score in zip(boxes, track_ids, confs):

                x1, y1, x2, y2 = map(int, box)

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Person box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(
                    frame,
                    f"ID {track_id} {score:.2f}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

                cv2.circle(frame, (center_x, center_y), 4, (255, 0, 0), -1)

                # Line crossing
                if track_id in previous_positions:
                    previous_x = previous_positions[track_id]

                    # Left -> right = boarding
                    if previous_x < line_x <= center_x:
                        entered_ids.add(track_id)

                    # Right -> left = exiting/returning
                    elif previous_x > line_x >= center_x:
                        exited_ids.add(track_id)

                previous_positions[track_id] = center_x

                # Only people visible in THIS frame, still left of the
                # bus-entry line, count as currently waiting.
                if center_x < line_x and track_id not in entered_ids:
                    waiting_now += 1

        max_waiting_count = max(max_waiting_count, waiting_now)

        entered_count = len(entered_ids)
        exited_count = len(exited_ids)
        inside_count = max(0, entered_count - exited_count)

        # Dashboard overlay
        cv2.rectangle(frame, (10, 10), (310, 145), (0, 0, 0), -1)

        overlay_lines = [
            (f"Waiting: {waiting_now}", (255, 255, 255)),
            (f"Boarded: {entered_count}", (0, 255, 0)),
            (f"Exited: {exited_count}", (0, 255, 255)),
            (f"Net entered: {inside_count}", (255, 255, 255)),
        ]

        for i, (text, color) in enumerate(overlay_lines):
            cv2.putText(
                frame,
                text,
                (20, 40 + i * 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )

        writer.write(frame)

        if progress_callback and total_frames > 0:
            progress_callback(min(1.0, frame_index / total_frames), frame_index, total_frames)

    cap.release()
    writer.release()

    final_path = _encode_browser_friendly(raw_path, output_path)

    return (
        str(final_path),
        len(entered_ids),
        len(exited_ids),
        max_waiting_count
    )



# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🚌 SmartTransit AI")
st.sidebar.caption(
    "AI-powered public transport crowd monitoring"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Route Search",
        "YOLO Detection",
        "Bus Boarding Analysis",
        "Crowd Prediction",
        "ETM Data",
        "Historical Analytics"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Pipeline:\n"
    "YOLO + Tracking → Crowd/Boarding → "
    "ETM + Context → ML Prediction → Risk Alert"
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        '<p class="main-title">AI Public Transport Crowd Monitoring</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">'
        'Real-time bus-stop crowd monitoring and future overcrowding prediction'
        '</p>',
        unsafe_allow_html=True
    )

    latest = df.iloc[-1]

    avg_occupancy = (
        df["current_onboard"] / df["capacity"] * 100
    ).mean()

    high_count = int(
        (df["occupancy_after_stop"] >= 80).sum()
    )

    avg_waiting = int(df["waiting_crowd"].mean())

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "🚌 Buses",
        df["bus_id"].nunique()
    )

    c2.metric(
        "👥 Avg. Onboard",
        int(df["current_onboard"].mean())
    )

    c3.metric(
        "🚏 Avg. Waiting",
        avg_waiting
    )

    c4.metric(
        "⚠️ High-Crowd Records",
        high_count
    )

    st.write("")

    left, right = st.columns([1.5, 1])

    with left:

        st.subheader("📈 Occupancy Trend")

        trend = df.tail(150).copy()

        trend["occupancy"] = (
            trend["current_onboard"]
            / trend["capacity"] * 100
        )

        fig = px.line(
            trend,
            x="date",
            y="occupancy",
            color="bus_id",
            labels={
                "occupancy": "Occupancy (%)",
                "date": "Date"
            }
        )

        fig.add_hline(
            y=80,
            line_dash="dash",
            annotation_text="High crowd"
        )

        fig.add_hline(
            y=100,
            line_dash="dash",
            annotation_text="Capacity"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with right:

        st.subheader("🚦 Latest Bus Status")

        live = (
            df.sort_values("date")
            .groupby("bus_id")
            .tail(1)
            .copy()
        )

        live["occupancy"] = (
            live["current_onboard"]
            / live["capacity"] * 100
        )

        for _, r in live.iterrows():

            level = risk_level(r["occupancy"])

            st.markdown(
                f"**{r['bus_id']}** · {r['route']}  \n"
                f"{risk_icon(level)} **{level}** — "
                f"{int(r['current_onboard'])}/"
                f"{int(r['capacity'])} passengers "
                f"({r['occupancy']:.0f}%)"
            )

            st.progress(
                min(float(r["occupancy"]) / 100, 1.0)
            )

    st.subheader("🔄 How SmartTransit AI Works")

    a, b, c, d = st.columns(4)

    a.info(
        "📹 **YOLO + Tracking**\n\n"
        "Detects people and tracks unique IDs from bus-stop video."
    )

    b.info(
        "🚌 **Passenger Flow**\n\n"
        "Measures waiting crowd and people boarding the bus."
    )

    c.info(
        "🎫 **ETM + Context**\n\n"
        "Combines historical passenger, route and operational information."
    )

    d.info(
        "🤖 **ML Prediction**\n\n"
        "Predicts future passenger load and overcrowding risk."
    )


# ============================================================
# ROUTE SEARCH & LOW-CROWD RECOMMENDATION
# ============================================================

elif page == "Route Search":

    st.markdown(
        '<p class="main-title">🔍 Bus Route Search</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="subtitle">'
        'Find buses on your route and see which ones have the least crowd'
        '</p>',
        unsafe_allow_html=True
    )

    # Build From / To city lists from the route column (e.g. "Kurnool-Hyderabad")
    route_pairs = (
        df["route"].dropna().unique().tolist()
    )
    cities = sorted(
        {c for r in route_pairs for c in r.split("-")}
    )

    # --------------------------------------------------------
    # Search controls
    # NOTE: only ONE search mode runs at a time. Streamlit re-runs the whole
    # script on every interaction, so if both the From/To filter and the
    # quick-text filter ran unconditionally, whichever ran last would always
    # silently overwrite `results` with its (possibly stale) value. Using an
    # explicit mode switch avoids that.
    # --------------------------------------------------------
    search_mode = st.radio(
        "Search by",
        ["🧭 From / To", "⌨️ Quick text search"],
        horizontal=True
    )

    results = pd.DataFrame()

    if search_mode == "🧭 From / To":

        s1, s2, s3 = st.columns([1, 1, 1])

        with s1:
            source = st.selectbox("From", ["Any"] + cities, index=0)
        with s2:
            destination = st.selectbox("To", ["Any"] + cities, index=0)
        with s3:
            bus_type_filter = st.selectbox(
                "Bus type",
                ["Any"] + sorted(df["bus_type"].dropna().unique().tolist())
            )

        if source != "Any" or destination != "Any" or bus_type_filter != "Any":
            filtered = df.copy()

            if source != "Any":
                filtered = filtered[
                    filtered["route"].str.split("-").str[0] == source
                ]
            if destination != "Any":
                filtered = filtered[
                    filtered["route"].str.split("-").str[1] == destination
                ]
            if bus_type_filter != "Any":
                filtered = filtered[filtered["bus_type"] == bus_type_filter]

            results = filtered

            if results.empty:
                st.warning(
                    "No direct buses found for that combination. "
                    "Try setting one of the fields to 'Any', or switch to "
                    "quick text search."
                )
        else:
            st.caption("Pick a From city, To city, or bus type to search.")

    else:  # Quick text search
        query = st.text_input(
            "Search route, bus stop, bus type or bus ID",
            placeholder="e.g. Hyderabad, Market Stop, Volvo AC, AP0001"
        )
        if query:
            q = query.strip().lower()
            mask = (
                df["route"].str.lower().str.contains(q, na=False)
                | df["bus_stop"].str.lower().str.contains(q, na=False)
                | df["bus_type"].str.lower().str.contains(q, na=False)
                | df["bus_id"].str.lower().str.contains(q, na=False)
            )
            results = df[mask]
            if results.empty:
                st.warning("No buses match that search term.")

    # --------------------------------------------------------
    # Results + low-crowd recommendation
    # --------------------------------------------------------
    if not results.empty:

        # Crowd score: prefer occupancy_after_stop, fall back to current occupancy
        results = results.copy()
        if "occupancy_after_stop" in results.columns:
            results["crowd_score"] = results["occupancy_after_stop"]
        else:
            results["crowd_score"] = (
                results["current_onboard"] / results["capacity"] * 100
            )

        results["crowd_tag"] = results["crowd_score"].apply(risk_level)

        st.divider()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🚌 Buses found", len(results))
        m2.metric("📉 Avg. crowd", f"{results['crowd_score'].mean():.0f}%")
        m3.metric(
            "🟢 Low-crowd buses",
            int((results["crowd_tag"] == "LOW").sum())
        )
        m4.metric(
            "🔴 Overcritical buses",
            int((results["crowd_tag"] == "OVERCRITICAL").sum())
        )

        # ---- Recommended (least crowded) buses ----
        st.subheader("✅ Recommended — Least Crowded Buses")

        top_recs = results.sort_values("crowd_score").head(5)

        if top_recs.empty:
            st.info("No buses match your search yet.")
        else:
            rec_cols = st.columns(len(top_recs))
            for col, (_, r) in zip(rec_cols, top_recs.iterrows()):
                with col:
                    st.markdown(
                        f"**{r['bus_id']}**  \n"
                        f"{r['route']}  \n"
                        f"{r['bus_type']}"
                    )
                    st.markdown(
                        f"{risk_icon(r['crowd_tag'])} **{r['crowd_tag']}** "
                        f"— {r['crowd_score']:.0f}% full"
                    )
                    st.progress(min(float(r["crowd_score"]) / 100, 1.0))
                    if "next_bus_arrival_min" in r and pd.notna(r["next_bus_arrival_min"]):
                        st.caption(
                            f"🕐 Next arrival in {int(r['next_bus_arrival_min'])} min"
                        )

        # ---- Full results table ----
        st.subheader("📋 All Matching Buses")

        sort_choice = st.radio(
            "Sort by",
            ["Least crowded first", "Most crowded first", "Earliest arrival"],
            horizontal=True
        )

        table = results.copy()
        if sort_choice == "Least crowded first":
            table = table.sort_values("crowd_score")
        elif sort_choice == "Most crowded first":
            table = table.sort_values("crowd_score", ascending=False)
        elif sort_choice == "Earliest arrival" and "next_bus_arrival_min" in table.columns:
            table = table.sort_values("next_bus_arrival_min")

        display_cols = [
            c for c in [
                "bus_id", "route", "bus_type", "bus_stop",
                "current_onboard", "capacity", "crowd_score",
                "crowd_tag", "next_bus_arrival_min", "bus_frequency_min"
            ] if c in table.columns
        ]

        st.dataframe(
            table[display_cols].rename(columns={
                "crowd_score": "crowd_%",
                "crowd_tag": "crowd_level",
                "next_bus_arrival_min": "arrival_min",
                "bus_frequency_min": "frequency_min"
            }),
            use_container_width=True,
            hide_index=True
        )

        # ---- Chart ----
        fig = px.bar(
            table.head(30),
            x="bus_id",
            y="crowd_score",
            color="crowd_tag",
            color_discrete_map={
                "LOW": "#22c55e",
                "MODERATE": "#eab308",
                "HIGH": "#f97316",
                "OVERCRITICAL": "#ef4444"
            },
            labels={"crowd_score": "Crowd (%)", "bus_id": "Bus"}
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info(
            "👆 Choose a From/To route above, or use the quick text search, "
            "to find buses and get low-crowd recommendations."
        )


# ============================================================
# YOLO DETECTION
# ============================================================

elif page == "YOLO Detection":

    st.title("📹 YOLO Person Detection & Tracking")

    st.write(
        "Upload your bus-stop video. The system detects people, "
        "assigns tracking IDs and identifies movement across the "
        "bus-entry line."
    )

    uploaded = st.file_uploader(
        "Upload bus-stop video",
        type=["mp4", "avi", "mov"]
    )

    col1, col2 = st.columns(2)

    with col1:
        confidence = st.slider(
            "YOLO confidence",
            0.20,
            0.80,
            0.35,
            0.05
        )

    with col2:
        line_percent = st.slider(
            "Bus-entry line position (%)",
            60,
            95,
            84
        )

    st.caption(
        "For your supplied video, the bus is on the right side. "
        "Start around 84% and adjust if required."
    )

    if uploaded:

        st.video(uploaded)

        if st.button(
            "▶️ Run YOLO + ByteTrack",
            type="primary"
        ):

            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def _on_progress(pct, frame_index, total_frames):
                progress_bar.progress(pct)
                status_text.caption(
                    f"Processing frame {frame_index} of {total_frames} "
                    f"({pct * 100:.0f}%)"
                )

            with st.spinner(
                "Running person detection and tracking..."
            ):

                try:

                    output_path, entered, exited, max_waiting = process_video(
                        uploaded.getvalue(),
                        confidence=confidence,
                        line_ratio=line_percent / 100,
                        progress_callback=_on_progress
                    )

                    progress_bar.progress(1.0)
                    status_text.empty()

                    st.success(
                        "Video processing completed."
                    )

                    output_bytes = Path(output_path).read_bytes()

                    st.video(output_bytes, format="video/mp4")

                    st.download_button(
                        "⬇️ Download detected video",
                        data=output_bytes,
                        file_name="smarttransit_detection.mp4",
                        mime="video/mp4"
                    )

                    c1, c2, c3 = st.columns(3)

                    c1.metric(
                        "People Boarded",
                        entered
                    )

                    c2.metric(
                        "People Crossed Back",
                        exited
                    )

                    c3.metric(
                        "Maximum Waiting Count",
                        max_waiting
                    )

                    st.info(
                        "The boarding count is generated from "
                        "tracked person IDs crossing the bus-entry line."
                    )

                except Exception as e:

                    st.error(
                        f"YOLO processing failed: {e}"
                    )

                    st.code(
                        "pip install ultralytics opencv-python"
                    )

    else:

        st.info(
            "Upload your bus-stop video to start real detection."
        )


# ============================================================
# BUS BOARDING ANALYSIS
# ============================================================

elif page == "Bus Boarding Analysis":

    st.title("🚌 Bus Boarding & Remaining Crowd")

    st.write(
        "This page connects the computer-vision boarding count "
        "with the bus capacity and waiting crowd."
    )

    st.subheader("Enter the detected values")

    c1, c2, c3 = st.columns(3)

    with c1:
        waiting = st.number_input(
            "People waiting at bus stop",
            min_value=0,
            max_value=500,
            value=20
        )

    with c2:
        boarding = st.number_input(
            "People boarded",
            min_value=0,
            max_value=500,
            value=7
        )

    with c3:
        capacity = st.number_input(
            "Bus capacity",
            min_value=1,
            max_value=200,
            value=50
        )

    remaining = max(0, waiting - boarding)

    st.write("")

    a, b, c = st.columns(3)

    a.metric(
        "🚏 Initial Waiting Crowd",
        waiting
    )

    b.metric(
        "🚌 Boarded",
        boarding
    )

    c.metric(
        "👥 Remaining Crowd",
        remaining
    )

    if waiting > 0:

        boarding_percentage = (
            boarding / waiting * 100
        )

        st.progress(
            min(boarding_percentage / 100, 1.0)
        )

        st.write(
            f"Boarding completion: "
            f"**{boarding_percentage:.1f}%**"
        )

    st.subheader("Project Logic")

    st.code("""
Waiting Crowd
      -
People Boarded
      =
Remaining Bus-Stop Crowd

Example:

20 waiting
 7 boarded
-----------
13 remaining
""")

    st.warning(
        "For the final prototype, waiting and boarding values "
        "should come directly from the YOLO tracking module."
    )


# ============================================================
# CROWD PREDICTION
# ============================================================

elif page == "Crowd Prediction":

    st.title("🤖 Future Crowd Prediction")

    st.write(
        "Random Forest predicts the future passenger load using "
        "historical transport and passenger-flow features."
    )

    st.caption(
        f"Model validation on the prototype dataset: "
        f"MAE = {model_mae:.2f} passengers | "
        f"R² = {model_r2:.3f}"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        current = st.number_input(
            "Current onboard passengers",
            0,
            200,
            35
        )

        waiting = st.number_input(
            "Waiting crowd",
            0,
            500,
            15
        )

        boarding = st.number_input(
            "Boarding count",
            0,
            100,
            8
        )

        alighting = st.number_input(
            "Alighting count",
            0,
            100,
            3
        )

    with c2:

        hour = st.slider(
            "Hour",
            0,
            23,
            8
        )

        day = st.selectbox(
            "Day",
            list(range(7)),
            format_func=lambda x: [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ][x]
        )

        holiday = st.checkbox(
            "Holiday"
        )

        event = st.checkbox(
            "Major event"
        )

    with c3:

        capacity = st.selectbox(
            "Bus capacity",
            sorted(df["capacity"].dropna().unique().astype(int).tolist())
        )

        route = st.selectbox(
            "Route",
            sorted(df["route"].dropna().unique().tolist())
        )

        bus = st.selectbox(
            "Bus",
            sorted(df["bus_id"].dropna().unique().tolist())
        )

        distance = st.number_input(
            "Distance (km)",
            min_value=0.0,
            max_value=1000.0,
            value=float(df["distance_km"].median())
        )

        next_bus = st.number_input(
            "Next bus arrival (min)",
            0,
            120,
            10
        )

        frequency = st.selectbox(
            "Bus frequency (min)",
            [5, 10, 15, 20, 30],
            index=1
        )

    if st.button(
        "🔮 Predict Future Crowd",
        type="primary"
    ):

        predicted = predict_future(
            current_onboard=current,
            waiting_crowd=waiting,
            boarding=boarding,
            alighting=alighting,
            capacity=capacity,
            distance=distance,
            hour=hour,
            weekend=day >= 5,
            holiday=holiday,
            event_flag=event,
            next_bus_arrival=next_bus,
            bus_frequency=frequency
        )

        occupancy = (
            predicted / capacity * 100
        )

        level = risk_level(occupancy)

        st.write("")

        a, b, c = st.columns(3)

        a.metric(
            "Predicted Passengers",
            predicted
        )

        b.metric(
            "Predicted Occupancy",
            f"{occupancy:.1f}%"
        )

        c.metric(
            "Risk Level",
            f"{risk_icon(level)} {level}"
        )

        if level == "OVERCRITICAL":

            st.error(
                f"🚨 {bus} on {route} is predicted to exceed "
                "safe operating capacity."
            )

        elif level == "HIGH":

            st.warning(
                f"⚠️ {bus} on {route} is predicted to have "
                "high crowding."
            )

        elif level == "MODERATE":

            st.info(
                "🟡 Moderate crowd expected. Continue monitoring."
            )

        else:

            st.success(
                "🟢 Crowd is expected to remain within a "
                "lower-risk range."
            )

        remaining = max(0, waiting - boarding)

        st.subheader("Passenger Flow Summary")

        st.dataframe(
            pd.DataFrame([{
                "Bus": bus,
                "Route": route,
                "Current Onboard": current,
                "Waiting Crowd": waiting,
                "Boarding": boarding,
                "Alighting": alighting,
                "Remaining Waiting": remaining,
                "Capacity": capacity,
                "Predicted Future": predicted,
                "Predicted Occupancy %": round(occupancy, 2),
                "Risk": level
            }]),
            use_container_width=True
        )


# ============================================================
# ETM DATA
# ============================================================

elif page == "ETM Data":

    st.title("🎫 ETM / Historical Passenger Data")

    st.write(
        "Historical passenger and operational data used by "
        "the prediction pipeline."
    )

    cols = [
        "date",
        "bus_id",
        "route",
        "capacity",
        "current_onboard",
        "boarding_count",
        "alighting_count",
        "passengers_after_stop",
        "occupancy_after_stop"
    ]

    available = [c for c in cols if c in df.columns]

    st.dataframe(
        df.tail(100)[available],
        use_container_width=True
    )

    st.subheader("ETM Attributes Used")

    st.markdown("""
    - Bus/service ID
    - Route ID
    - Passenger count
    - Boarding count
    - Alighting count
    - Capacity
    - Date/time
    - Stop information
    - Historical passenger flow
    """)

    st.info(
        "In deployment, authorized ETM data can replace the "
        "prototype CSV records."
    )


# ============================================================
# HISTORICAL ANALYTICS
# ============================================================

elif page == "Historical Analytics":

    st.title("📊 Historical Crowd Analytics")

    c1, c2 = st.columns(2)

    with c1:

        route = st.selectbox(
            "Select route",
            ["All"] + sorted(
                df["route"].dropna().unique().tolist()
            )
        )

    with c2:

        bus_type = st.selectbox(
            "Select bus type",
            ["All"] + sorted(
                df["bus_type"].dropna().unique().tolist()
            )
        )

    filtered = df.copy()

    if route != "All":
        filtered = filtered[
            filtered["route"] == route
        ]

    if bus_type != "All":
        filtered = filtered[
            filtered["bus_type"] == bus_type
        ]

    st.metric(
        "Records",
        len(filtered)
    )

    left, right = st.columns(2)

    with left:

        fig1 = px.histogram(
            filtered,
            x="occupancy_after_stop",
            nbins=20,
            labels={
                "occupancy_after_stop":
                "Occupancy After Stop (%)"
            }
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    with right:

        grouped = (
            filtered.groupby("route", as_index=False)
            ["future_passengers"]
            .mean()
            .sort_values("future_passengers", ascending=False)
        )

        fig2 = px.bar(
            grouped,
            x="route",
            y="future_passengers",
            labels={
                "future_passengers":
                "Average Future Passengers"
            }
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    st.subheader("Crowd Level Distribution")

    level_counts = (
        filtered["crowd_level"]
        .value_counts()
        .reset_index()
    )

    level_counts.columns = [
        "Crowd Level",
        "Records"
    ]

    fig3 = px.pie(
        level_counts,
        names="Crowd Level",
        values="Records"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        filtered.head(100),
        use_container_width=True
    )


# ============================================================
# SYSTEM ARCHITECTURE
# ============================================================

else:

    st.title("🏗️ SmartTransit AI System Architecture")

    st.code("""
                    BUS STOP CAMERA
                           │
                           ▼
                 ┌──────────────────┐
                 │ YOLO PERSON      │
                 │ DETECTION        │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ BYTE TRACK /     │
                 │ BOBoT-SORT       │
                 │ PERSON TRACKING  │
                 └────────┬─────────┘
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
          WAITING ZONE         BUS ENTRY ZONE
                │                   │
                ▼                   ▼
        Waiting Crowd         Boarding Count
                │                   │
                └─────────┬─────────┘
                          ▼
                  PASSENGER FLOW
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
           ETM          ROUTE         TIME
           DATA          DATA        / DAY
             │            │            │
             └────────────┼────────────┘
                          ▼
                  FEATURE ENGINEERING
                          │
                          ▼
                  RANDOM FOREST ML
                          │
                          ▼
                 FUTURE PASSENGERS
                          │
                          ▼
                   OCCUPANCY %
                          │
                  ┌───────┴────────┐
                  ▼                ▼
               RISK LEVEL       ALERT
            LOW/MODERATE/HIGH   SYSTEM
                          │
                          ▼
                   STREAMLIT UI
    """, language="text")

    st.subheader("Technology Stack")

    st.markdown("""
    **Computer Vision:** YOLO / Ultralytics + OpenCV  
    **Tracking:** ByteTrack  
    **Machine Learning:** Scikit-learn Random Forest  
    **Data Processing:** Pandas + NumPy  
    **Visualization:** Plotly  
    **Dashboard:** Streamlit  
    **Prototype Data:** Updated SmartTransit CSV  
    **Future Data Sources:** ETM + GPS + authorized transport feeds
    """)

    st.subheader("Important Prototype Limitation")

    st.warning(
        "The newly added stop-level vision/boarding fields in the "
        "prototype CSV are simulated placeholders. During actual "
        "operation, YOLO will replace those values with real "
        "computer-vision measurements."
    )

    st.success(
        "Core concept: detect the present passenger situation "
        "with computer vision and use historical/contextual data "
        "to predict future overcrowding."
    )

