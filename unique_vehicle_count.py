from ultralytics import YOLO
from pathlib import Path

# --------------------------------------------------
# TrafficGuardAI - Unique Vehicle Counting
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_DIR = BASE_DIR / "data" / "videos"

# Load YOLO model
model = YOLO("yolo11n.pt")

# Find MP4 videos
videos = list(VIDEO_DIR.glob("*.mp4"))

if not videos:
    print("❌ No MP4 video found!")
    print(f"Put a video inside: {VIDEO_DIR}")
    exit()

video = videos[0]

print("✅ Found video:", video.name)
print("🎥 Tracking unique vehicles...")
print("Please wait...")

# Sets store unique tracking IDs
unique_cars = set()
unique_motorcycles = set()
unique_buses = set()
unique_trucks = set()

# Track video
results = model.track(
    source=str(video),
    conf=0.25,
    stream=True,
    persist=True,
    verbose=False
)

for result in results:

    if result.boxes.id is None:
        continue

    tracking_ids = result.boxes.id.int().cpu().tolist()
    class_ids = result.boxes.cls.int().cpu().tolist()

    for tracking_id, class_id in zip(tracking_ids, class_ids):

        class_name = model.names[class_id]

        if class_name == "car":
            unique_cars.add(tracking_id)

        elif class_name == "motorcycle":
            unique_motorcycles.add(tracking_id)

        elif class_name == "bus":
            unique_buses.add(tracking_id)

        elif class_name == "truck":
            unique_trucks.add(tracking_id)


# Calculate total
total_unique_vehicles = (
    len(unique_cars)
    + len(unique_motorcycles)
    + len(unique_buses)
    + len(unique_trucks)
)

# Display results
print()
print("=" * 50)
print("🚦 TRAFFICGUARD AI - UNIQUE VEHICLES")
print("=" * 50)

print("🚗 Unique Cars:          ", len(unique_cars))
print("🏍️ Unique Motorcycles:   ", len(unique_motorcycles))
print("🚌 Unique Buses:         ", len(unique_buses))
print("🚚 Unique Trucks:        ", len(unique_trucks))

print("-" * 50)

print("🚘 TOTAL UNIQUE VEHICLES:", total_unique_vehicles)

print("=" * 50)
print("✅ Unique vehicle counting completed!")