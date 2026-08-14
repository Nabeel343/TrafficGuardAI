from ultralytics import YOLO
from pathlib import Path

# --------------------------------------------------
# TrafficGuardAI - Video Detection
# --------------------------------------------------

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_DIR = BASE_DIR / "data" / "videos"
OUTPUT_DIR = BASE_DIR / "data" / "output"

# Create output folder
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find video files
video_files = []

for extension in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
    video_files.extend(VIDEO_DIR.glob(extension))

if not video_files:
    print("❌ No video found!")
    print(f"Please put a traffic video inside: {VIDEO_DIR}")
    exit()

print(f"✅ Found {len(video_files)} video(s)")

# Load YOLO model
model = YOLO("yolo11n.pt")

# Process each video
for video in video_files:

    print(f"\n🎥 Processing: {video.name}")

    results = model.predict(
        source=str(video),
        conf=0.25,
        save=True,
        project=str(OUTPUT_DIR),
        name="video_detections",
        exist_ok=True,
        show=False
    )

    print(f"✅ Finished: {video.name}")

print("\n" + "=" * 45)
print("🚦 TRAFFICGUARD AI VIDEO DETECTION COMPLETE")
print("=" * 45)

print(f"📁 Processed video saved in:")
print(OUTPUT_DIR / "video_detections")