import streamlit as st
from pathlib import Path
from ultralytics import YOLO
import cv2
import tempfile

# ============================================================
# TRAFFICGUARD AI
# REAL-TIME TRAFFIC ANALYTICS DASHBOARD
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "dashboard_result.mp4"

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TrafficGuard AI",
    page_icon="🚦",
    layout="wide"
)

# ============================================================
# HEADER
# ============================================================

st.title("🚦 TrafficGuard AI")

st.subheader(
    "AI-Powered Traffic Monitoring & Analytics"
)

st.markdown("---")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ TrafficGuard AI")

st.sidebar.info(
    """
    Upload a traffic video and TrafficGuard AI will:

    🚗 Detect vehicles
    🏍️ Classify vehicle types
    🔢 Track vehicles
    📊 Calculate traffic density
    🚦 Estimate congestion
    🎥 Generate processed video
    """
)

# ============================================================
# VIDEO UPLOAD
# ============================================================

st.header("📤 Upload Traffic Video")

uploaded_file = st.file_uploader(
    "Choose a traffic video",
    type=["mp4", "avi", "mov", "mkv"]
)

# ============================================================
# ANALYSIS BUTTON
# ============================================================

if uploaded_file is not None:

    st.video(uploaded_file)

    analyze = st.button(
        "🚀 Analyze Traffic",
        type="primary",
        use_container_width=True
    )

    if analyze:

        # ----------------------------------------------------
        # Save uploaded video temporarily
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as temp_file:

            temp_file.write(
                uploaded_file.read()
            )

            input_video = temp_file.name

        st.info(
            "🤖 TrafficGuard AI is analyzing the video..."
        )

        progress = st.progress(0)

        # ----------------------------------------------------
        # Load YOLO11
        # ----------------------------------------------------

        model = YOLO("yolo11n.pt")

        # ----------------------------------------------------
        # Open video
        # ----------------------------------------------------

        cap = cv2.VideoCapture(input_video)

        if not cap.isOpened():

            st.error(
                "❌ Could not open the uploaded video."
            )

            st.stop()

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 30

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        # ----------------------------------------------------
        # Video writer
        # ----------------------------------------------------

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        out = cv2.VideoWriter(
            str(OUTPUT_VIDEO),
            fourcc,
            fps,
            (width, height)
        )

        # ----------------------------------------------------
        # Vehicle classes
        # ----------------------------------------------------

        vehicle_classes = {
            2: "Car",
            3: "Motorcycle",
            5: "Bus",
            7: "Truck"
        }

        # ----------------------------------------------------
        # Tracking sets
        # ----------------------------------------------------

        unique_ids = set()

        car_ids = set()
        motorcycle_ids = set()
        bus_ids = set()
        truck_ids = set()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_vehicle_detections = 0
        max_vehicles = 0

        frame_count = 0

        # ----------------------------------------------------
        # Process video
        # ----------------------------------------------------

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # YOLO tracking
            results = model.track(
                frame,
                persist=True,
                conf=0.25,
                verbose=False
            )

            cars = 0
            motorcycles = 0
            buses = 0
            trucks = 0

            # ------------------------------------------------
            # Process detections
            # ------------------------------------------------

            for result in results:

                if result.boxes is None:
                    continue

                for box in result.boxes:

                    class_id = int(
                        box.cls[0]
                    )

                    if class_id not in vehicle_classes:
                        continue

                    vehicle_type = vehicle_classes[
                        class_id
                    ]

                    # Count current frame
                    if class_id == 2:
                        cars += 1

                    elif class_id == 3:
                        motorcycles += 1

                    elif class_id == 5:
                        buses += 1

                    elif class_id == 7:
                        trucks += 1

                    # Tracking ID
                    if box.id is None:
                        continue

                    track_id = int(
                        box.id[0]
                    )

                    unique_ids.add(track_id)

                    # Store by type
                    if class_id == 2:
                        car_ids.add(track_id)

                    elif class_id == 3:
                        motorcycle_ids.add(track_id)

                    elif class_id == 5:
                        bus_ids.add(track_id)

                    elif class_id == 7:
                        truck_ids.add(track_id)

                    # Bounding box
                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    confidence = float(
                        box.conf[0]
                    )

                    label = (
                        f"{vehicle_type} "
                        f"#{track_id} "
                        f"{confidence:.2f}"
                    )

                    # Draw box
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # Draw label
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

            # ------------------------------------------------
            # Frame statistics
            # ------------------------------------------------

            vehicles_in_frame = (
                cars +
                motorcycles +
                buses +
                trucks
            )

            total_vehicle_detections += (
                vehicles_in_frame
            )

            max_vehicles = max(
                max_vehicles,
                vehicles_in_frame
            )

            # ------------------------------------------------
            # Average density
            # ------------------------------------------------

            average_vehicles = (
                total_vehicle_detections /
                frame_count
            )

            # ------------------------------------------------
            # Congestion
            # ------------------------------------------------

            if average_vehicles < 2:

                traffic_level = "LOW"

            elif average_vehicles < 4:

                traffic_level = "MEDIUM"

            else:

                traffic_level = "HIGH"

            # ------------------------------------------------
            # Information panel
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (10, 10),
                (330, 190),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                "TRAFFICGUARD AI",
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Cars: {cars}",
                (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Motorcycles: {motorcycles}",
                (20, 94),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Buses: {buses}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Trucks: {trucks}",
                (20, 146),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Traffic: {traffic_level}",
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            # Write frame
            out.write(frame)

            # Progress
            if total_frames > 0:

                progress_value = (
                    frame_count /
                    total_frames
                )

                progress.progress(
                    min(progress_value, 1.0)
                )

        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        cap.release()
        out.release()

        progress.progress(1.0)

        # ----------------------------------------------------
        # Final statistics
        # ----------------------------------------------------

        if frame_count > 0:

            average_vehicles = (
                total_vehicle_detections /
                frame_count
            )

        else:

            average_vehicles = 0

        # ----------------------------------------------------
        # Final traffic level
        # ----------------------------------------------------

        if average_vehicles < 2:

            traffic_level = "LOW"

        elif average_vehicles < 4:

            traffic_level = "MEDIUM"

        else:

            traffic_level = "HIGH"

        # ====================================================
        # RESULTS
        # ====================================================

        st.success(
            "✅ Traffic analysis completed!"
        )

        st.markdown("---")

        st.header("📊 Traffic Analytics")

        # ----------------------------------------------------
        # Vehicle metrics
        # ----------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "🚗 Cars",
                len(car_ids)
            )

        with col2:

            st.metric(
                "🏍️ Motorcycles",
                len(motorcycle_ids)
            )

        with col3:

            st.metric(
                "🚌 Buses",
                len(bus_ids)
            )

        with col4:

            st.metric(
                "🚚 Trucks",
                len(truck_ids)
            )

        # ----------------------------------------------------
        # Traffic metrics
        # ----------------------------------------------------

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "🚘 Unique Vehicles",
                len(unique_ids)
            )

        with col2:

            st.metric(
                "📊 Average / Frame",
                f"{average_vehicles:.2f}"
            )

        with col3:

            st.metric(
                "🔝 Peak / Frame",
                max_vehicles
            )

        # ----------------------------------------------------
        # Traffic status
        # ----------------------------------------------------

        st.markdown("---")

        st.header("🚦 Traffic Status")

        if traffic_level == "LOW":

            st.success(
                "🟢 LOW TRAFFIC"
            )

        elif traffic_level == "MEDIUM":

            st.warning(
                "🟡 MEDIUM TRAFFIC"
            )

        else:

            st.error(
                "🔴 HIGH TRAFFIC"
            )

        # ----------------------------------------------------
        # Vehicle distribution
        # ----------------------------------------------------

        st.markdown("---")

        st.header(
            "🚘 Vehicle Distribution"
        )

        distribution = {

            "Cars": len(car_ids),

            "Motorcycles": len(
                motorcycle_ids
            ),

            "Buses": len(
                bus_ids
            ),

            "Trucks": len(
                truck_ids
            )
        }

        st.bar_chart(
            distribution
        )

        # ----------------------------------------------------
        # Processed video
        # ----------------------------------------------------

        st.markdown("---")

        st.header(
            "🎥 Processed Traffic Video"
        )

        if OUTPUT_VIDEO.exists():

            video_bytes = (
                OUTPUT_VIDEO.read_bytes()
            )

            st.video(
                video_bytes
            )

            st.download_button(
                label="📥 Download Processed Video",
                data=video_bytes,
                file_name="trafficguard_ai_result.mp4",
                mime="video/mp4",
                use_container_width=True
            )

        else:

            st.error(
                "❌ Processed video was not created."
            )

# ============================================================
# NO VIDEO
# ============================================================

else:

    st.info(
        "👆 Upload a traffic video above to begin analysis."
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "TrafficGuard AI • YOLO11 • Computer Vision • "
    "Vehicle Tracking • Traffic Analytics"
)