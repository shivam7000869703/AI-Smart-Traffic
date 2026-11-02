

import streamlit as st
import cv2
from ultralytics import YOLO
import pandas as pd
import tempfile
import os
import torch
import numpy as np

# ---------------------------------------------------------
# AI SMART TRAFFIC MANAGEMENT SYSTEM
# 4-SIDE INTERSECTION ANALYSIS
# NORTH + SOUTH + EAST + WEST
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Smart Traffic Management",
    page_icon="🚦",
    layout="wide",
)

# ---------------------------------------------------------
# CSS
# ---------------------------------------------------------

st.markdown("""
<style>
/* Main page background */
.stApp {
    background: linear-gradient(135deg, #0B172A 0%, #10263D 52%, #142F49 100%);
    color: #F4F8FC;
}

/* This app uses a top navigation bar instead of a sidebar */
[data-testid="stSidebar"] {
    display: none;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
    max-width: 1500px;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #1B3854, #183149);
    padding: 16px 18px;
    border-radius: 13px;
    border: 1px solid #315573;
    box-shadow: 0 5px 18px rgba(0, 0, 0, 0.14);
}
[data-testid="stMetricLabel"] {
    color: #C6D9E9 !important;
}
[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
}
[data-testid="stMetricDelta"] {
    color: #7FE0B0 !important;
}

/* Top navigation radio group */
div[role="radiogroup"] {
    background: #172F48;
    border: 1px solid #31516D;
    border-radius: 12px;
    padding: 10px 14px;
    gap: 14px;
}
div[role="radiogroup"] label {
    color: #EAF4FC !important;
    font-weight: 600;
}

/* Upload areas and tables */
[data-testid="stFileUploader"] section {
    background: #172D43;
    border: 1px dashed #4B789B;
    border-radius: 12px;
}
[data-testid="stDataFrame"] {
    border: 1px solid #31516D;
    border-radius: 10px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #0878D1, #1596C8);
    color: #FFFFFF;
    border: 0;
    border-radius: 9px;
    font-weight: 700;
    min-height: 42px;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #0868B5, #087EAC);
    color: #FFFFFF;
    border: 0;
}

/* Text and headings */
h1, h2, h3 {
    color: #F7FBFF;
}
p, li, label {
    color: #DCE8F2;
}
hr {
    border-color: #31516D;
}

/* Intro hero panel */
.hero-panel {
    background: linear-gradient(115deg, #183B5B, #176B87);
    border: 1px solid #367A98;
    border-radius: 16px;
    padding: 24px 26px;
    margin: 14px 0 20px 0;
    box-shadow: 0 8px 24px rgba(0,0,0,.18);
}
.hero-panel h2 {
    color: #FFFFFF;
    margin: 0 0 8px 0;
}
.hero-panel p {
    color: #E4F3FC;
    font-size: 15px;
    margin-bottom: 0;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

VEHICLE_CLASSES = {
    2: "Cars",
    3: "Motorcycles",
    5: "Buses",
    7: "Trucks"
}

SIDES = ["North", "South", "East", "West"]


# ---------------------------------------------------------
# LOAD YOLO MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model():

    model = YOLO("yolo11n.pt")

    return model


# ---------------------------------------------------------
# VIDEO INFO
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_video_info(path):

    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    cap.release()

    duration = frames / fps if fps else 0

    return {
        "fps": fps,
        "frames": frames,
        "width": width,
        "height": height,
        "duration": duration
    }


# ---------------------------------------------------------
# TRAFFIC DENSITY
# ---------------------------------------------------------

def calculate_density(average):

    if average > 15:

        return "HIGH", 60

    elif average > 5:

        return "MEDIUM", 40

    else:

        return "LOW", 20


# ---------------------------------------------------------
# ANALYZE ONE SIDE
# ---------------------------------------------------------

def run_analysis(video_path):

    model = load_model()

    # CPU / GPU automatically selected
    if torch.cuda.is_available():
        device = 0
    else:
        device = "cpu"

    results = model.track(
        source=video_path,
        stream=True,

        device=device,

        # Improved detection settings
        imgsz=512,
        conf=0.20,
        iou=0.45,

        # Allow more detections
        max_det=300,

        # Analyze every frame
        vid_stride=1,

        tracker="bytetrack.yaml",

        verbose=False
    )

    # Unique tracking IDs
    unique_ids = {
        2: set(),
        3: set(),
        5: set(),
        7: set()
    }

    total_current = 0

    frame_count = 0

    maximum = 0

    low = 0
    medium = 0
    high = 0

    # -----------------------------------------------------
    # PROCESS VIDEO
    # -----------------------------------------------------

    for result in results:

        if result.boxes is None:
            continue

        if result.boxes.cls is None:
            continue

        classes = (
            result.boxes.cls
            .int()
            .cpu()
            .tolist()
        )

        # Current vehicles in frame
        current = sum(
            1 for c in classes
            if c in VEHICLE_CLASSES
        )

        frame_count += 1

        total_current += current

        maximum = max(
            maximum,
            current
        )

        # Density frames
        if current <= 5:

            low += 1

        elif current <= 15:

            medium += 1

        else:

            high += 1

        # Tracking IDs
        if result.boxes.id is not None:

            ids = (
                result.boxes.id
                .int()
                .cpu()
                .tolist()
            )

            for tid, cls in zip(
                ids,
                classes
            ):

                if cls in unique_ids:

                    unique_ids[cls].add(tid)

    # -----------------------------------------------------
    # CALCULATE AVERAGE
    # -----------------------------------------------------

    average = (
        total_current / frame_count
        if frame_count
        else 0
    )

    density, green_time = calculate_density(
        average
    )

    # -----------------------------------------------------
    # RETURN RESULTS
    # -----------------------------------------------------

    return {

        "cars":
            len(unique_ids[2]),

        "motorcycles":
            len(unique_ids[3]),

        "buses":
            len(unique_ids[5]),

        "trucks":
            len(unique_ids[7]),

        "total_unique":
            sum(
                len(v)
                for v in unique_ids.values()
            ),

        "average_per_frame":
            average,

        "maximum_per_frame":
            maximum,

        "low_frames":
            low,

        "medium_frames":
            medium,

        "high_frames":
            high,

        "density":
            density,

        "green_time":
            green_time,

        "frames_processed":
            frame_count
    }




# ---------------------------------------------------------
# IMAGE ANALYSIS - ONE DIRECTION
# ---------------------------------------------------------

def run_image_analysis(image):

    model = load_model()

    if torch.cuda.is_available():
        device = 0
    else:
        device = "cpu"

    result = model.predict(
        source=image,
        device=device,
        imgsz=512,
        conf=0.20,
        iou=0.45,
        max_det=300,
        verbose=False
    )[0]

    counts = {
        2: 0,  # Cars
        3: 0,  # Motorcycles
        5: 0,  # Buses
        7: 0   # Trucks
    }

    if result.boxes is not None and result.boxes.cls is not None:
        classes = (
            result.boxes.cls
            .int()
            .cpu()
            .tolist()
        )

        for cls in classes:
            if cls in counts:
                counts[cls] += 1

    total = sum(counts.values())

    # For a single image, density is based on vehicles visible in that image.
    density, green_time = calculate_density(total)

    annotated = result.plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    return {
        "cars": counts[2],
        "motorcycles": counts[3],
        "buses": counts[5],
        "trucks": counts[7],
        "total": total,
        "density": density,
        "green_time": green_time,
        "annotated": annotated
    }


# ---------------------------------------------------------
# SAVE UPLOADED VIDEO
# ---------------------------------------------------------

def save_uploaded_video(uploaded_file):

    suffix = os.path.splitext(
        uploaded_file.name
    )[1]

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    )

    temp_file.write(
        uploaded_file.read()
    )

    temp_file.close()

    return temp_file.name


# ---------------------------------------------------------
# TOP NAVIGATION BAR (replaces the sidebar)
# ---------------------------------------------------------

nav_brand, nav_menu = st.columns([1.25, 3.2], vertical_alignment="center")

with nav_brand:
    st.markdown("### 🚦 AI SMART TRAFFIC")

with nav_menu:
    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Video Analysis",
            "Image Analysis",
            "IoT Integration",
            "System Information"
        ],
        horizontal=True,
        label_visibility="collapsed",
        key="top_navigation"
    )

st.markdown("---")


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.title(
        "🚦 AI-Based Traffic Management System"
    )

    st.caption(
        "Computer Vision + AI + 4-Side Traffic Intersection Management"
    )

    # -----------------------------------------------------
    # PROJECT INTRODUCTION / DASHBOARD OVERVIEW
    # -----------------------------------------------------

    st.markdown("""
    <div class="hero-panel">
        <h2>Welcome to Smart Traffic Management</h2>
        <p>
            An AI-powered traffic monitoring system designed to detect,
            track and count vehicles from all four directions of an
            intersection. The system compares traffic conditions and
            recommends green-signal durations to support smoother traffic flow.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("⚙️ System Overview")
    overview_1, overview_2, overview_3, overview_4 = st.columns(4)
    overview_1.metric("Road Directions", "4", "North • South • East • West")
    overview_2.metric("Detection Model", "YOLO11", "Vehicle detection")
    overview_3.metric("Tracking System", "ByteTrack", "Vehicle ID tracking")
    overview_4.metric("Signal Logic", "Adaptive", "Based on analyzed traffic")

    st.caption(
        "Workflow: Upload four traffic videos → Detect and track vehicles → "
        "Compare traffic density → Recommend signal timing."
    )

    # -----------------------------------------------------
    # UPLOAD SECTION
    # -----------------------------------------------------

    st.markdown("---")

    st.subheader(
        "📹 Upload Traffic Videos"
    )

    st.write(
        "Upload one traffic video for each direction of the intersection."
    )

    n_col, s_col, e_col, w_col = st.columns(4)

    with n_col:

        north_file = st.file_uploader(
            "🟢 North Traffic",
            type=[
                "mp4",
                "avi",
                "mov",
                "mkv"
            ],
            key="north_video"
        )

    with s_col:

        south_file = st.file_uploader(
            "🔵 South Traffic",
            type=[
                "mp4",
                "avi",
                "mov",
                "mkv"
            ],
            key="south_video"
        )

    with e_col:

        east_file = st.file_uploader(
            "🟠 East Traffic",
            type=[
                "mp4",
                "avi",
                "mov",
                "mkv"
            ],
            key="east_video"
        )

    with w_col:

        west_file = st.file_uploader(
            "🔴 West Traffic",
            type=[
                "mp4",
                "avi",
                "mov",
                "mkv"
            ],
            key="west_video"
        )

    uploaded_videos = {

        "North": north_file,

        "South": south_file,

        "East": east_file,

        "West": west_file
    }

    st.write("")

    analyze_button = st.button(
        "🚦 Analyze All Four Sides",
        type="primary",
        use_container_width=True
    )

    # -----------------------------------------------------
    # RUN FOUR SIDE ANALYSIS
    # -----------------------------------------------------

    if analyze_button:

        missing = [
            side
            for side, file in uploaded_videos.items()
            if file is None
        ]

        if missing:

            st.warning(
                "Please upload videos for: "
                + ", ".join(missing)
            )

        else:

            side_results = {}

            progress = st.progress(0)

            status_text = st.empty()

            for index, side in enumerate(SIDES):

                status_text.info(
                    f"🔍 Analyzing {side} traffic..."
                )

                uploaded_file = uploaded_videos[
                    side
                ]

                video_path = save_uploaded_video(
                    uploaded_file
                )

                try:

                    result = run_analysis(
                        video_path
                    )

                    side_results[side] = result

                finally:

                    if os.path.exists(
                        video_path
                    ):

                        os.remove(
                            video_path
                        )

                progress.progress(
                    (index + 1) / 4
                )

            st.session_state[
                "side_results"
            ] = side_results

            status_text.success(
                "✅ All four sides analyzed successfully!"
            )

    # -----------------------------------------------------
    # RESULTS
    # -----------------------------------------------------

    side_results = st.session_state.get(
        "side_results",
        None
    )

    if side_results is None:

        st.info(
            "📹 Upload North, South, East and West traffic videos, then click "
            "'Analyze All Four Sides'."
        )

        st.stop()

    # -----------------------------------------------------
    # FOUR SIDE SUMMARY
    # -----------------------------------------------------

    st.markdown("---")

    st.subheader(
        "📊 Four-Side Traffic Summary"
    )

    # Four direction metrics
    c1, c2, c3, c4 = st.columns(4)

    for col, side in zip(
        [c1, c2, c3, c4],
        SIDES
    ):

        result = side_results[side]

        density = result["density"]

        if density == "LOW":

            symbol = "🟢"

        elif density == "MEDIUM":

            symbol = "🟡"

        else:

            symbol = "🔴"

        col.metric(
            f"{symbol} {side}",
            f'{result["total_unique"]} Unique Vehicles',
            f'{density} | {result["green_time"]} sec'
        )

    # -----------------------------------------------------
    # VEHICLE TYPE TOTALS
    # -----------------------------------------------------

    st.markdown("---")

    st.subheader(
        "🚗 Vehicle Analysis by Direction"
    )

    comparison_df = pd.DataFrame({

        "Direction": SIDES,

        "Cars": [
            side_results[s]["cars"]
            for s in SIDES
        ],

        "Motorcycles": [
            side_results[s]["motorcycles"]
            for s in SIDES
        ],

        "Buses": [
            side_results[s]["buses"]
            for s in SIDES
        ],

        "Trucks": [
            side_results[s]["trucks"]
            for s in SIDES
        ],

        "Unique Vehicles (Video Total)": [
            side_results[s]["total_unique"]
            for s in SIDES
        ],

        "Average / Frame": [
            round(
                side_results[s]["average_per_frame"],
                2
            )
            for s in SIDES
        ],

        "Density": [
            side_results[s]["density"]
            for s in SIDES
        ],

        "Green Time": [
            side_results[s]["green_time"]
            for s in SIDES
        ]
    })

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------------------------------
    # VEHICLE COUNT CARDS
    # -----------------------------------------------------

    st.markdown("---")

    st.subheader(
        "🚘 Vehicle Counts"
    )

    for side in SIDES:

        result = side_results[side]

        st.markdown(
            f"### {'🟢' if side == 'North' else '🔵' if side == 'South' else '🟠' if side == 'East' else '🔴'} {side}"
        )

        a, b, c, d, e = st.columns(5)

        a.metric(
            "🚗 Cars",
            result["cars"]
        )

        b.metric(
            "🏍️ Motorcycles",
            result["motorcycles"]
        )

        c.metric(
            "🚌 Buses",
            result["buses"]
        )

        d.metric(
            "🚚 Trucks",
            result["trucks"]
        )

        e.metric(
            "🚘 Unique Vehicles",
            result["total_unique"]
        )

    # -----------------------------------------------------
    # TRAFFIC DENSITY GRAPH
    # -----------------------------------------------------

    st.markdown("---")

    graph_col, signal_col = st.columns(
        [1.4, 1]
    )

    with graph_col:

        st.subheader(
            "📊 Traffic Comparison"
        )

        traffic_chart = pd.DataFrame({

            "Direction": SIDES,

            "Vehicles": [
                side_results[s][
                    "total_unique"
                ]
                for s in SIDES
            ]
        })

        st.bar_chart(
            traffic_chart.set_index(
                "Direction"
            )
        )

    # -----------------------------------------------------
    # SIGNAL DECISION
    # -----------------------------------------------------

    with signal_col:

        st.subheader(
            "🚦 Signal Timing"
        )

        for side in SIDES:

            result = side_results[side]

            if result["density"] == "HIGH":

                icon = "🔴"

            elif result["density"] == "MEDIUM":

                icon = "🟡"

            else:

                icon = "🟢"

            st.write(
                f"{icon} **{side}:** "
                f"{result['green_time']} seconds"
            )

    # -----------------------------------------------------
    # AI DECISION
    # -----------------------------------------------------

    st.markdown("---")

    st.subheader(
        "🧠 AI Traffic Decision"
    )

    # Find highest traffic side
    highest_side = max(
        SIDES,
        key=lambda s:
        side_results[s][
            "average_per_frame"
        ]
    )

    highest_result = side_results[
        highest_side
    ]

    st.info(
        f"""
        **Highest Traffic Direction:** {highest_side}

        The AI detected an average of
        **{highest_result["average_per_frame"]:.2f} vehicles/frame**
        on the {highest_side} side.

        Traffic Density:
        **{highest_result["density"]}**

        Recommended Green Time:
        **{highest_result["green_time"]} seconds**

        The traffic controller can give higher priority
        to the {highest_side} direction while maintaining
        signal cycles for the other directions.
        """
    )

    # -----------------------------------------------------
    # SIGNAL SEQUENCE
    # -----------------------------------------------------

    st.markdown("---")

    st.subheader(
        "🔄 Recommended Signal Sequence"
    )

    # Sort directions according to traffic
    sorted_sides = sorted(
        SIDES,
        key=lambda s:
        side_results[s][
            "average_per_frame"
        ],
        reverse=True
    )

    sequence_text = " → ".join(
        [
            f"{side} ({side_results[side]['green_time']}s)"
            for side in sorted_sides
        ]
    )

    st.success(
        f"🚦 {sequence_text}"
    )


# =========================================================
# VIDEO ANALYSIS PAGE
# =========================================================

elif page == "Video Analysis":

    st.title(
        "🎥 Four-Side Computer Vision Analysis"
    )

    side_results = st.session_state.get(
        "side_results",
        None
    )

    if side_results is None:

        st.warning(
            "Please analyze the four traffic videos from the Dashboard first."
        )

    else:

        for side in SIDES:

            result = side_results[side]

            st.markdown(
                f"## {side} Traffic"
            )

            a, b, c, d, e = st.columns(5)

            a.metric(
                "Cars",
                result["cars"]
            )

            b.metric(
                "Motorcycles",
                result["motorcycles"]
            )

            c.metric(
                "Buses",
                result["buses"]
            )

            d.metric(
                "Trucks",
                result["trucks"]
            )

            e.metric(
                "Total",
                result["total_unique"]
            )

            st.write(
                f"Density: **{result['density']}**"
            )

            st.write(
                f"Recommended Green Time: "
                f"**{result['green_time']} seconds**"
            )

            st.markdown("---")





# =========================================================
# IMAGE ANALYSIS PAGE
# =========================================================

elif page == "Image Analysis":

    st.title("🖼️ Four-Side Image Analysis")

    st.caption(
        "Upload one traffic image for each direction "
        "of the intersection."
    )

    st.markdown("---")
    st.subheader("📸 Upload Traffic Images")

    n_col, s_col, e_col, w_col = st.columns(4)

    with n_col:
        north_image = st.file_uploader(
            "🟢 North Image",
            type=["jpg", "jpeg", "png", "webp"],
            key="north_image"
        )

    with s_col:
        south_image = st.file_uploader(
            "🔵 South Image",
            type=["jpg", "jpeg", "png", "webp"],
            key="south_image"
        )

    with e_col:
        east_image = st.file_uploader(
            "🟠 East Image",
            type=["jpg", "jpeg", "png", "webp"],
            key="east_image"
        )

    with w_col:
        west_image = st.file_uploader(
            "🔴 West Image",
            type=["jpg", "jpeg", "png", "webp"],
            key="west_image"
        )

    uploaded_images = {
        "North": north_image,
        "South": south_image,
        "East": east_image,
        "West": west_image
    }

    st.markdown("---")

    analyze_images = st.button(
        "🖼️ Analyze All Four Images",
        type="primary",
        use_container_width=True
    )

    if analyze_images:

        missing = [
            side
            for side, file in uploaded_images.items()
            if file is None
        ]

        if missing:
            st.warning(
                "Please upload images for: "
                + ", ".join(missing)
            )

        else:
            image_results = {}
            progress = st.progress(0)
            status = st.empty()

            for index, side in enumerate(SIDES):

                status.info(f"🔍 Analyzing {side} image...")

                image_bytes = uploaded_images[side].getvalue()

                image = cv2.imdecode(
                    np.frombuffer(image_bytes, dtype=np.uint8),
                    cv2.IMREAD_COLOR
                )

                if image is None:
                    st.error(f"Could not read {side} image.")
                    st.stop()

                image_results[side] = run_image_analysis(image)

                progress.progress((index + 1) / 4)

            st.session_state["image_results"] = image_results

            status.success(
                "✅ All four images analyzed successfully!"
            )

    # -----------------------------------------------------
    # IMAGE ANALYSIS RESULTS
    # -----------------------------------------------------

    image_results = st.session_state.get("image_results")

    if image_results is None:

        st.info(
            "Upload North, South, East and West images, then click "
            "'Analyze All Four Images'."
        )

    else:

        st.markdown("---")
        st.subheader("📊 Four-Side Image Summary")

        c1, c2, c3, c4 = st.columns(4)

        for col, side in zip([c1, c2, c3, c4], SIDES):

            result = image_results[side]

            col.metric(
                f"{side}",
                f'{result["total"]} Vehicles',
                f'{result["density"]} | {result["green_time"]} sec'
            )

        # -------------------------------------------------
        # VEHICLE COMPARISON TABLE
        # -------------------------------------------------

        st.markdown("---")
        st.subheader("🚗 Vehicle Analysis by Direction")

        image_df = pd.DataFrame({
            "Direction": SIDES,
            "Cars": [image_results[s]["cars"] for s in SIDES],
            "Motorcycles": [image_results[s]["motorcycles"] for s in SIDES],
            "Buses": [image_results[s]["buses"] for s in SIDES],
            "Trucks": [image_results[s]["trucks"] for s in SIDES],
            "Total Vehicles": [image_results[s]["total"] for s in SIDES],
            "Density": [image_results[s]["density"] for s in SIDES],
            "Green Time": [image_results[s]["green_time"] for s in SIDES]
        })

        st.dataframe(
            image_df,
            use_container_width=True,
            hide_index=True
        )

        # -------------------------------------------------
        # DETECTED IMAGE PREVIEWS
        # -------------------------------------------------

        st.markdown("---")
        st.subheader("🔍 Vehicle Detection Results")

        image_columns = st.columns(2)

        for index, side in enumerate(SIDES):

            result = image_results[side]

            with image_columns[index % 2]:

                st.markdown(f"### {side} Traffic")

                st.image(
                    result["annotated"],
                    caption=f"{side} - YOLO11 Vehicle Detection",
                    use_container_width=True
                )

                a, b = st.columns(2)
                a.metric("🚘 Total Vehicles", result["total"])
                b.metric("🚦 Density", result["density"])

                st.write(
                    f"Cars: **{result['cars']}** · "
                    f"Motorcycles: **{result['motorcycles']}**"
                )
                st.write(
                    f"Buses: **{result['buses']}** · "
                    f"Trucks: **{result['trucks']}**"
                )
                st.write(
                    f"Recommended Green Time: "
                    f"**{result['green_time']} seconds**"
                )

        # -------------------------------------------------
        # TRAFFIC COMPARISON GRAPH
        # -------------------------------------------------

        st.markdown("---")
        st.subheader("📊 Four-Side Traffic Comparison")

        chart_df = pd.DataFrame({
            "Direction": SIDES,
            "Vehicles": [image_results[s]["total"] for s in SIDES]
        })

        st.bar_chart(chart_df.set_index("Direction"))

        # -------------------------------------------------
        # AI TRAFFIC DECISION
        # -------------------------------------------------

        st.markdown("---")
        st.subheader("🧠 AI Traffic Decision")

        highest_side = max(
            SIDES,
            key=lambda s: image_results[s]["total"]
        )

        highest_result = image_results[highest_side]

        st.info(
            f"""
            **Highest Traffic Direction:** {highest_side}

            Detected Vehicles: **{highest_result['total']}**

            Traffic Density: **{highest_result['density']}**

            Recommended Green Time:
            **{highest_result['green_time']} seconds**
            """
        )

        st.caption(
            "Image Analysis counts vehicles visible in each still image. "
            "ByteTrack tracking IDs are used only in Video Analysis."
        )


# =========================================================
# IOT INTEGRATION PAGE
# =========================================================

elif page == "IoT Integration":

    st.title("🔌 IoT Device Integration")
    st.caption("Register and manage IoT devices for your smart traffic management system.")

    st.markdown("""
    <div class="hero-panel">
        <h2>Connect Your Traffic IoT Devices</h2>
        <p>Add devices such as ESP32, Arduino, sensors, or an IoT traffic controller.
        Device registration is stored in this Streamlit session. Live hardware control
        requires the device's communication endpoint and integration code.</p>
    </div>
    """, unsafe_allow_html=True)

    if "iot_devices" not in st.session_state:
        st.session_state["iot_devices"] = []

    with st.form("iot_device_form", clear_on_submit=True):
        st.subheader("➕ Add a New IoT Device")

        device_col1, device_col2 = st.columns(2)
        with device_col1:
            device_name = st.text_input(
                "Device Name",
                placeholder="e.g. North Road Traffic Sensor"
            )
            device_type = st.selectbox(
                "Device Type",
                [
                    "Traffic Sensor",
                    "ESP32",
                    "Arduino",
                    "Raspberry Pi",
                    "Traffic Light Controller",
                    "Camera",
                    "Other"
                ]
            )
            device_direction = st.selectbox(
                "Intersection Direction",
                ["North", "South", "East", "West", "Center / Common"]
            )

        with device_col2:
            device_id = st.text_input(
                "Unique Device ID",
                placeholder="e.g. ESP32-NORTH-001"
            )
            protocol = st.selectbox(
                "Communication Protocol",
                ["MQTT", "HTTP / REST", "Wi-Fi", "Bluetooth", "Serial / USB", "Other"]
            )
            endpoint = st.text_input(
                "Broker / IP Address / Endpoint (optional)",
                placeholder="e.g. mqtt://192.168.1.10:1883"
            )

        submitted_device = st.form_submit_button(
            "➕ Add IoT Device",
            type="primary",
            use_container_width=True
        )

        if submitted_device:
            if not device_name.strip() or not device_id.strip():
                st.error("Please enter both Device Name and Unique Device ID.")
            elif any(
                d["device_id"].casefold() == device_id.strip().casefold()
                for d in st.session_state["iot_devices"]
            ):
                st.error("A device with this ID is already registered.")
            else:
                st.session_state["iot_devices"].append({
                    "name": device_name.strip(),
                    "device_type": device_type,
                    "device_id": device_id.strip(),
                    "direction": device_direction,
                    "protocol": protocol,
                    "endpoint": endpoint.strip() or "Not configured",
                    "status": "Registered (not connected)"
                })
                st.success(f"Device '{device_name.strip()}' added successfully.")

    st.markdown("---")
    st.subheader("📋 Registered IoT Devices")

    devices = st.session_state["iot_devices"]
    if not devices:
        st.info("No IoT devices added yet. Use the form above to register your first device.")
    else:
        device_df = pd.DataFrame(devices)
        device_df = device_df.rename(columns={
            "name": "Device Name",
            "device_type": "Device Type",
            "device_id": "Device ID",
            "direction": "Direction",
            "protocol": "Protocol",
            "endpoint": "Endpoint",
            "status": "Connection Status"
        })
        st.dataframe(device_df, use_container_width=True, hide_index=True)

        device_to_remove = st.selectbox(
            "Select a device to remove",
            options=["Select device"] + [
                f'{d["device_id"]} — {d["name"]}' for d in devices
            ],
            key="iot_remove_device"
        )
        if st.button("🗑️ Remove Selected Device", key="iot_remove_button"):
            if device_to_remove != "Select device":
                selected_id = device_to_remove.split(" — ")[0]
                st.session_state["iot_devices"] = [
                    d for d in st.session_state["iot_devices"]
                    if d["device_id"] != selected_id
                ]
                st.success(f"Device {selected_id} removed.")
                st.rerun()

    st.markdown("---")
    st.subheader("🧪 Connection Setup Guide")
    st.markdown("""
    1. Register your device above and select its communication protocol.
    2. For MQTT, configure an MQTT broker address and topic on the device.
    3. Add the appropriate MQTT/HTTP/serial client code to receive sensor data
       or send commands from this application.
    4. Test using a non-production device before connecting to any physical
       traffic signal controller.
    """)
    st.warning(
        "Safety note: this page currently registers devices only; it does not "
        "establish a live connection or operate physical traffic lights. "
        "Existing video/image analysis and signal recommendations remain unchanged."
    )


# =========================================================
# SYSTEM INFORMATION PAGE
# =========================================================

elif page == "System Information":

    st.title("ℹ️ System Information")
    st.caption("Technical overview of the four-side AI traffic management prototype")

    st.markdown("""
    <div class="hero-panel">
        <h2>About the Project</h2>
        <p>
            The AI-Based Traffic Management System analyzes four separate
            traffic videos representing the North, South, East and West
            approaches of an intersection. It detects vehicles, tracks them
            across video frames, estimates traffic density and recommends
            green-signal durations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    info_left, info_right = st.columns(2)

    with info_left:
        st.subheader("🧠 AI and Computer Vision")
        st.markdown("""
        - **YOLO11n:** detects vehicles in video frames.
        - **ByteTrack:** associates detections across frames using tracking IDs.
        - **OpenCV:** reads and processes the uploaded videos.
        - **Vehicle classes:** cars, motorcycles, buses and trucks.
        """)

    with info_right:
        st.subheader("🚦 Traffic Management")
        st.markdown("""
        - Each uploaded video is analyzed independently for its direction.
        - Unique tracked vehicles and average vehicles per processed frame are calculated.
        - Traffic density is classified from the average vehicles per frame.
        - Green-time recommendations are generated from the density category.
        """)

    st.subheader("🔄 Processing Workflow")
    st.code("""Four Direction Videos (North / South / East / West)
                    ↓
              OpenCV Video Input
                    ↓
             YOLO11 Vehicle Detection
                    ↓
              ByteTrack Tracking
                    ↓
          Vehicle Counting by Category
                    ↓
        Average Vehicles per Processed Frame
                    ↓
             Traffic Density
                    ↓
        Recommended Green Signal Time""", language="text")

    st.subheader("📌 Current Prototype Signal Rules")
    rules_df = pd.DataFrame({
        "Average vehicles / frame": ["0 to 5", "Above 5 to 15", "Above 15"],
        "Density": ["LOW", "MEDIUM", "HIGH"],
        "Recommended green time": ["20 seconds", "40 seconds", "60 seconds"]
    })
    st.dataframe(rules_df, use_container_width=True, hide_index=True)

    st.info(
        "This is a software prototype. The dashboard recommends signal timing; "
        "physical traffic lights are not controlled unless a separate IoT/controller "
        "module is connected and implemented."
    )
