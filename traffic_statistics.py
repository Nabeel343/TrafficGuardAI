
from ultralytics import YOLO
from pathlib import Path

# --------------------------------------------------
# TrafficGuardAI - Traffic Statistics
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_DIR = BASE_DIR / "data" / "input"

INPUT_VIDEO = VIDEO_DIR / "traffic.mp4.mp4"

# --------------------------------------------------
# Check video
# --------------------------------------------------

if not INPUT_VIDEO.exists():

    print("❌ Video not found!")
    print(f"Expected video at: {INPUT_VIDEO}")
    exit()

print("=" * 55)
print("🚦 TRAFFICGUARD AI - TRAFFIC STATISTICS")
print("=" * 55)

print(f"🎥 Analyzing: {INPUT_VIDEO.name}")
print("Please wait...\n")

# --------------------------------------------------
# Load YOLO model
# --------------------------------------------------

model = YOLO("yolo11n.pt")

# --------------------------------------------------
# Counters
# --------------------------------------------------

total_frames = 0

max_vehicles = 0

total_vehicle_detections = 0

cars = 0
motorcycles = 0
buses = 0
trucks = 0
persons = 0

# --------------------------------------------------
# Vehicle classes
# --------------------------------------------------

vehicle_classes = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# --------------------------------------------------
# Analyze video
# --------------------------------------------------

results = model.predict(
    source=str(INPUT_VIDEO),
    conf=0.25,
    stream=True,
    verbose=False
)

for result in results:

    total_frames += 1

    vehicles_in_frame = 0

    for box in result.boxes:

        class_id = int(box.cls[0])

        # Person
        if class_id == 0:

            persons += 1

            continue

        # Ignore non-vehicle classes
        if class_id not in vehicle_classes:

            continue

        vehicles_in_frame += 1

        # Count vehicle type
        if class_id == 2:

            cars += 1

        elif class_id == 3:

            motorcycles += 1

        elif class_id == 5:

            buses += 1

        elif class_id == 7:

            trucks += 1

    # Total vehicle detections
    total_vehicle_detections += vehicles_in_frame

    # Maximum vehicles in a frame
    if vehicles_in_frame > max_vehicles:

        max_vehicles = vehicles_in_frame


# --------------------------------------------------
# Calculate average
# --------------------------------------------------

if total_frames > 0:

    average_vehicles = (
        total_vehicle_detections / total_frames
    )

else:

    average_vehicles = 0


# --------------------------------------------------
# Traffic level
# --------------------------------------------------

if max_vehicles <= 3:

    traffic_level = "LOW"

elif max_vehicles <= 6:

    traffic_level = "MEDIUM"

else:

    traffic_level = "HIGH"


# --------------------------------------------------
# Vehicle totals
# --------------------------------------------------

total_detections = (
    cars +
    motorcycles +
    buses +
    trucks
)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print()
print("=" * 55)
print("📊 TRAFFICGUARD AI - ANALYTICS REPORT")
print("=" * 55)

print()
print("🎥 VIDEO ANALYSIS")
print("-" * 40)

print(f"Total Frames:              {total_frames}")
print(f"Maximum Vehicles/Frame:    {max_vehicles}")
print(f"Average Vehicles/Frame:    {average_vehicles:.2f}")

print()
print("🚘 VEHICLE DETECTIONS")
print("-" * 40)

print(f"🚗 Cars:                   {cars}")
print(f"🏍️ Motorcycles:            {motorcycles}")
print(f"🚌 Buses:                  {buses}")
print(f"🚚 Trucks:                 {trucks}")

print(f"🚘 Total Vehicle Detections: {total_detections}")

print()
print("👤 OTHER OBJECTS")
print("-" * 40)

print(f"👤 Persons:                {persons}")

print()
print("🚦 TRAFFIC STATUS")
print("-" * 40)

print(f"Traffic Level:             {traffic_level}")

print()
print("=" * 55)
print("✅ TRAFFIC STATISTICS COMPLETED")
print("=" * 55)
