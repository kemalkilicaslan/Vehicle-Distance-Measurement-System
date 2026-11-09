# Vehicle Distance Measurement System — Configuration

# Multiple ROI zones for a 2042 x 1148px frame — each is a list of [x, y] points
ROI_ZONES = {
    "LEFT":  [[240, 600], [925, 550], [312, 1100], [100, 1100]],
    "MAIN":  [[925, 550], [1025, 550], [1712, 1100], [312, 1100]],
    "RIGHT": [[1025, 550], [1802, 600], [1942, 1100], [1712, 1100]],
}

# Camera focal length (pixels) and per-zone optical center
FOCAL_LENGTH = 500
OPTICAL_CENTERS = {"LEFT": (156, 1050), "MAIN": (1000, 1020), "RIGHT": (1868, 1050)}

# Vehicle classes to detect (COCO IDs): car, motorcycle, bus, truck
TARGET_CLASSES = [2, 3, 5, 7]

# Real-world vehicle heights (meters) — class ID -> height
REAL_VEHICLE_HEIGHTS = {2: 1.55, 3: 1.2, 5: 3.0, 7: 2.5}

# Confidence thresholds
CONFIDENCE_THRESHOLD = 0.7
VEHICLE_PLATE_CONFIDENCE = 0.475

# Distance display / warning constraints
MAX_DISPLAY_DISTANCE = 15
WARNING_DISTANCES = {"LEFT": 1, "MAIN": 2, "RIGHT": 1}

# Per-ROI distance display threshold (distances above this are hidden)
DISTANCE_DISPLAY_THRESHOLDS = {
    "LEFT": 5,
    "RIGHT": 5,
}

# MAIN ROI warning threshold (distances below this are drawn red)
MAIN_ROI_WARNING_THRESHOLD = 5.0

# Model and video files
MODEL_PATH = "yolo12x.pt"
VEHICLE_PLATE_MODEL_PATH = "vehicle-plate.pt"
INPUT_VIDEO = "dashcam_video.mov"                  # input video file name
OUTPUT_VIDEO = "Vehicle-Distance-Measurement.mp4"  # output video file name
