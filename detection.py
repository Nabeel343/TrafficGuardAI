from ultralytics import YOLO
from pathlib import Path
from collections import Counter

# --------------------------------------------------
# TrafficGuardAI - Vehicle Detection & Counting
# --------------------------------------------------

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "data" / "input"
OUTPUT_DIR = BASE_DIR / "data" / "output"

# Create output folder
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load YOLO model
model = YOLO("yolo11n.pt")

# Find input images
image_files = []

for extension in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
    image_files.extend(INPUT_DIR.glob(extension))

if not image_files:
    print("❌ No images found!")
    print(f"Please put an image inside: {INPUT_DIR}")
    exit()

print(f"✅ Found {len(image_files)} image(s)")

# Vehicle classes
vehicle_classes = {
    "car",
    "motorcycle",
    "bus",
    "truck"
}

# Process each image
for image in image_files:

    print(f"\n🔍 Processing: {image.name}")

    results = model.predict(
        source=str(image),
        conf=0.25,
        save=True,
        project=str(OUTPUT_DIR),
        name="detections",
        exist_ok=True
    )

    # Get detected class names
    detected_objects = []

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            detected_objects.append(class_name)

    # Count objects
    counts = Counter(detected_objects)

    # Vehicle counts
    cars = counts.get("car", 0)
    motorcycles = counts.get("motorcycle", 0)
    buses = counts.get("bus", 0)
    trucks = counts.get("truck", 0)

    persons = counts.get("person", 0)

    total_vehicles = cars + motorcycles + buses + trucks

    # Traffic level
    if total_vehicles < 10:
        traffic_level = "LOW"
    elif total_vehicles < 20:
        traffic_level = "MEDIUM"
    else:
        traffic_level = "HIGH"

    # Display results
    print("\n" + "=" * 40)
    print("🚦 TRAFFICGUARD AI RESULTS")
    print("=" * 40)

    print(f"🚗 Cars:          {cars}")
    print(f"🏍️ Motorcycles:   {motorcycles}")
    print(f"🚌 Buses:         {buses}")
    print(f"🚚 Trucks:        {trucks}")
    print(f"👤 Persons:       {persons}")

    print("-" * 40)

    print(f"🚘 Total Vehicles: {total_vehicles}")
    print(f"🚦 Traffic Level:  {traffic_level}")

    print("=" * 40)


print("\n✅ TrafficGuardAI detection completed!")

print(
    f"📁 Results saved to: "
    f"{OUTPUT_DIR / 'detections'}"
)