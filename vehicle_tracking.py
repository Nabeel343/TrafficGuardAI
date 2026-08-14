
from ultralytics import YOLO
from pathlib import Path
import cv2

# --------------------------------------------------
# TrafficGuardAI - Vehicle Tracking
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

# Correct input folder
VIDEO_DIR = BASE_DIR / "data" / "input"

# Output folder
OUTPUT_DIR = BASE_DIR / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_VIDEO = VIDEO_DIR / "traffic.mp4.mp4"
OUTPUT_VIDEO = OUTPUT_DIR / "vehicle_tracking.mp4"

# --------------------------------------------------
# Check video
# --------------------------------------------------

if not INPUT_VIDEO.exists():
    print("❌ Video not found!")
    print(f"Expected video at: {INPUT_VIDEO}")
    exit()

print("=" * 55)
print("🚦 TRAFFICGUARD AI - VEHICLE TRACKING")
print("=" * 55)
print(f"🎥 Input: {INPUT_VIDEO.name}")
print("Please wait...\n")

# --------------------------------------------------
# Load YOLO model
# --------------------------------------------------

model = YOLO("yolo11n.pt")

# --------------------------------------------------
# Open video
# --------------------------------------------------

cap = cv2.VideoCapture(str(INPUT_VIDEO))

if not cap.isOpened():
    print("❌ Could not open video.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

# --------------------------------------------------
# Video writer
# --------------------------------------------------

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    fps,
    (width, height)
)

# --------------------------------------------------
# Vehicle classes
# --------------------------------------------------

vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

frame_count = 0
unique_ids = set()

car_ids = set()
motorcycle_ids = set()
bus_ids = set()
truck_ids = set()

print("🚗 Tracking vehicles...")
print("Please wait...\n")

# --------------------------------------------------
# Process video
# --------------------------------------------------

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

    for result in results:

        if result.boxes is None:
            continue

        boxes = result.boxes

        for box in boxes:

            class_id = int(box.cls[0])

            # Ignore non-vehicle objects
            if class_id not in vehicle_classes:
                continue

            vehicle_type = vehicle_classes[class_id]

            # Get tracking ID
            if box.id is None:
                continue

            track_id = int(box.id[0])

            unique_ids.add(track_id)

            # Store ID according to vehicle type
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

            confidence = float(box.conf[0])

            # Label
            label = f"{vehicle_type} #{track_id} {confidence:.2f}"

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Label background
            (text_width, text_height), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                2
            )

            cv2.rectangle(
                frame,
                (x1, max(0, y1 - text_height - 8)),
                (x1 + text_width + 5, y1),
                (0, 255, 0),
                -1
            )

            # Label text
            cv2.putText(
                frame,
                label,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )

    # --------------------------------------------------
    # Statistics panel
    # --------------------------------------------------

    cv2.rectangle(
        frame,
        (10, 10),
        (330, 155),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "TRAFFICGUARD AI",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Unique Vehicles: {len(unique_ids)}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Cars: {len(car_ids)}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Motorcycles: {len(motorcycle_ids)}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Buses: {len(bus_ids)}  Trucks: {len(truck_ids)}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # Write output
    out.write(frame)

# --------------------------------------------------
# Cleanup
# --------------------------------------------------

cap.release()
out.release()

# --------------------------------------------------
# Final results
# --------------------------------------------------

print("\n" + "=" * 55)
print("✅ VEHICLE TRACKING COMPLETED")
print("=" * 55)

print(f"🎞️ Frames analyzed: {frame_count}")

print("\n🚘 UNIQUE VEHICLES")
print("-" * 40)

print(f"🚗 Unique Cars:         {len(car_ids)}")
print(f"🏍️ Unique Motorcycles:  {len(motorcycle_ids)}")
print(f"🚌 Unique Buses:        {len(bus_ids)}")
print(f"🚚 Unique Trucks:       {len(truck_ids)}")

print("-" * 40)
print(f"🚘 Total Unique IDs:    {len(unique_ids)}")

print(f"\n📁 Output:")
print(OUTPUT_VIDEO)

print("=" * 55)
