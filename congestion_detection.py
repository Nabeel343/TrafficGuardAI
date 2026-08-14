
from ultralytics import YOLO
import cv2
import os

# ============================================
# TRAFFICGUARD AI
# SMART CONGESTION DETECTION
# ============================================

MODEL_PATH = "yolo11n.pt"
INPUT_VIDEO = "data/input/traffic.mp4.mp4"
OUTPUT_VIDEO = "data/output/congestion_detection.mp4"

# Load model
model = YOLO(MODEL_PATH)

# Open video
cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    print("❌ Could not open input video.")
    exit()

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

# Output directory
os.makedirs("data/output", exist_ok=True)

# Video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

# Vehicle classes
vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

frame_count = 0

# Statistics
total_vehicle_detections = 0
max_vehicles = 0

total_cars = 0
total_motorcycles = 0
total_buses = 0
total_trucks = 0

print("=" * 60)
print("🚦 TRAFFICGUARD AI")
print("🧠 SMART CONGESTION DETECTION")
print("=" * 60)
print("🎥 Analyzing traffic video...")
print("Please wait...\n")


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # YOLO detection
    results = model(frame, verbose=False)

    # Current frame counts
    cars = 0
    motorcycles = 0
    buses = 0
    trucks = 0

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if class_id not in vehicle_classes:
                continue

            vehicle_type = vehicle_classes[class_id]

            # Count vehicle type
            if class_id == 2:
                cars += 1
                total_cars += 1

            elif class_id == 3:
                motorcycles += 1
                total_motorcycles += 1

            elif class_id == 5:
                buses += 1
                total_buses += 1

            elif class_id == 7:
                trucks += 1
                total_trucks += 1

            # Bounding box
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Label
            label = f"{vehicle_type} {confidence:.2f}"

            # Bounding box
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
                0.55,
                2
            )

            cv2.rectangle(
                frame,
                (x1, max(0, y1 - text_height - 10)),
                (x1 + text_width + 5, y1),
                (0, 255, 0),
                -1
            )

            # Label
            cv2.putText(
                frame,
                label,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2
            )

    # Current frame total
    vehicle_count = (
        cars +
        motorcycles +
        buses +
        trucks
    )

    total_vehicle_detections += vehicle_count

    if vehicle_count > max_vehicles:
        max_vehicles = vehicle_count

    # ============================================
    # SMART CONGESTION
    # ============================================

    # Average so far
    average_vehicles = (
        total_vehicle_detections / frame_count
    )

    # Use average + current density
    if average_vehicles < 2.0 and vehicle_count <= 3:

        congestion = "LOW"

    elif average_vehicles < 4.0 and vehicle_count <= 6:

        congestion = "MEDIUM"

    else:

        congestion = "HIGH"

    # ============================================
    # DISPLAY PANEL
    # ============================================

    cv2.rectangle(
        frame,
        (10, 10),
        (330, 220),
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
        f"Cars: {cars}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Motorcycles: {motorcycles}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Buses: {buses}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Trucks: {trucks}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Vehicles: {vehicle_count}",
        (20, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Avg Density: {average_vehicles:.2f}",
        (20, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Traffic: {congestion}",
        (20, 215),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # Write frame
    out.write(frame)


# ============================================
# FINAL RESULTS
# ============================================

cap.release()
out.release()

if frame_count > 0:

    final_average = (
        total_vehicle_detections / frame_count
    )

else:

    final_average = 0


# Final congestion assessment
if final_average < 2.0:
    final_congestion = "LOW"

elif final_average < 4.0:
    final_congestion = "MEDIUM"

else:
    final_congestion = "HIGH"


print("\n" + "=" * 60)
print("✅ SMART CONGESTION ANALYSIS COMPLETED")
print("=" * 60)

print(f"🎞️ Frames analyzed: {frame_count}")

print("\n🚘 VEHICLE DETECTIONS")
print("-" * 40)

print(f"🚗 Cars:          {total_cars}")
print(f"🏍️ Motorcycles:   {total_motorcycles}")
print(f"🚌 Buses:         {total_buses}")
print(f"🚚 Trucks:        {total_trucks}")

print("-" * 40)

print(f"🚘 Peak vehicles/frame:    {max_vehicles}")
print(f"📊 Average vehicles/frame: {final_average:.2f}")

print("\n🚦 FINAL TRAFFIC STATUS")
print("-" * 40)

print(f"Traffic Level: {final_congestion}")

print("\n📁 Output:")
print(OUTPUT_VIDEO)

print("=" * 60)
