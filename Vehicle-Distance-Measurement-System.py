# Vehicle Distance Measurement System

# Import the necessary libraries
import cv2
import numpy as np
from ultralytics import YOLO
import config

# Define coordinates for multiple ROI zones (loaded from config.py)
ROI_ZONES = {name: np.array(pts, dtype=np.int32) for name, pts in config.ROI_ZONES.items()}

# Configuration parameters (loaded from config.py)
FOCAL_LENGTH = config.FOCAL_LENGTH
OPTICAL_CENTERS = config.OPTICAL_CENTERS
TARGET_CLASSES = config.TARGET_CLASSES  # Car, Motorcycle, Bus, Truck
REAL_VEHICLE_HEIGHTS = config.REAL_VEHICLE_HEIGHTS  # Car, Motorcycle, Bus, Truck
CONFIDENCE_THRESHOLD = config.CONFIDENCE_THRESHOLD
VEHICLE_PLATE_CONFIDENCE = config.VEHICLE_PLATE_CONFIDENCE
MAX_DISPLAY_DISTANCE = config.MAX_DISPLAY_DISTANCE
WARNING_DISTANCES = config.WARNING_DISTANCES

# Distance display thresholds for each ROI (loaded from config.py)
DISTANCE_DISPLAY_THRESHOLDS = config.DISTANCE_DISPLAY_THRESHOLDS

# MAIN ROI warning threshold (distances < 5m will be red) (loaded from config.py)
MAIN_ROI_WARNING_THRESHOLD = config.MAIN_ROI_WARNING_THRESHOLD

# Load YOLO models (paths loaded from config.py)
model = YOLO(config.MODEL_PATH)
vehicle_plate_model = YOLO(config.VEHICLE_PLATE_MODEL_PATH)

def is_point_in_roi(point, roi_coordinates):
    return cv2.pointPolygonTest(roi_coordinates, point, False) >= 0

def get_vehicle_roi_zone(center_point):
    for zone_name, roi_coords in ROI_ZONES.items():
        if is_point_in_roi(center_point, roi_coords):
            return zone_name
    return None

def calculate_distance(bbox, class_id, zone_name):
    if class_id not in REAL_VEHICLE_HEIGHTS or zone_name not in OPTICAL_CENTERS:
        return 0
        
    x1, y1, x2, y2 = bbox
    bbox_height = y2 - y1
    
    if bbox_height <= 0:
        return 0
        
    # Calculate displacement from optical center
    vehicle_center_x, vehicle_center_y = (x1 + x2) / 2, (y1 + y2) / 2
    optical_center_x, optical_center_y = OPTICAL_CENTERS[zone_name]
    displacement = np.sqrt((vehicle_center_x - optical_center_x)**2 + (vehicle_center_y - optical_center_y)**2)
    
    # Distance calculation with correction factor
    distance = (REAL_VEHICLE_HEIGHTS[class_id] * FOCAL_LENGTH) / bbox_height
    return distance * (1.0 + displacement * 0.0001)

def should_display_distance(distance, zone_name):
    """Check if distance should be displayed based on ROI-specific thresholds"""
    threshold = DISTANCE_DISPLAY_THRESHOLDS.get(zone_name, 999)
    return distance <= threshold

def get_distance_color(distance, zone_name):
    """Determine color for distance display based on zone-specific rules"""
    warning_distance = WARNING_DISTANCES.get(zone_name, 2.0)
    
    # For MAIN ROI, use red color if distance < 5m
    if zone_name == 'MAIN' and distance < MAIN_ROI_WARNING_THRESHOLD:
        return (0, 0, 255)  # Red
    # For all ROIs, use red if within warning distance
    elif distance < warning_distance:
        return (0, 0, 255)  # Red
    else:
        return (0, 255, 0)  # Green

def blur_vehicle_plates_in_vehicle(frame, vehicle_bbox):
    x1, y1, x2, y2 = map(int, vehicle_bbox)
    vehicle_region = frame[y1:y2, x1:x2]
    
    if vehicle_region.size == 0:
        return
    
    try:
        plate_results = vehicle_plate_model(vehicle_region, conf=VEHICLE_PLATE_CONFIDENCE, verbose=False)
        
        for plate_result in plate_results:
            if plate_result.boxes is not None:
                for plate_box in plate_result.boxes.xyxy.cpu().numpy():
                    px1, py1, px2, py2 = map(int, plate_box)
                    
                    # Convert to original frame coordinates
                    frame_height, frame_width = frame.shape[:2]
                    px1, py1 = max(0, min(px1 + x1, frame_width)), max(0, min(py1 + y1, frame_height))
                    px2, py2 = max(0, min(px2 + x1, frame_width)), max(0, min(py2 + y1, frame_height))
                    
                    if px2 > px1 and py2 > py1:
                        plate_region = frame[py1:py2, px1:px2]
                        if plate_region.size > 0:
                            blurred_plate = cv2.GaussianBlur(plate_region, (51, 51), 30)
                            frame[py1:py2, px1:px2] = blurred_plate
                            
    except Exception as e:
        print(f"Vehicle plate detection error: {e}")

def draw_distance_label(frame, bbox, distance, color, zone_name):
    x1, y1, x2, y2 = map(int, bbox)
    distance_text = f"{distance:.1f}m"
    warning_distance = WARNING_DISTANCES.get(zone_name, 2.0)
    
    # Draw warning overlay if too close (based on original warning distances)
    if distance < warning_distance:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    
    # Text positioning
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale, thickness = 1.2, 3
    (text_width, text_height), _ = cv2.getTextSize(distance_text, font, font_scale, thickness)
    
    frame_height, frame_width = frame.shape[:2]
    center_x = (x1 + x2) // 2
    text_x = max(10, min(center_x - text_width // 2, frame_width - text_width - 20))
    text_y = min(frame_height - 35, y2 + text_height + 20)
    
    # Background rectangle
    padding = 10
    bg_x1 = max(0, text_x - padding)
    bg_y1 = max(0, text_y - text_height - padding)
    bg_x2 = min(frame_width, text_x + text_width + padding)
    bg_y2 = min(frame_height, text_y + padding)
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), 2)
    
    # Draw text with outline
    cv2.putText(frame, distance_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, distance_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

def draw_warning_message(frame, message):
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale, thickness = 1.0, 3
    
    (text_width, text_height), _ = cv2.getTextSize(message, font, font_scale, thickness)
    position_x, position_y = width//2 - text_width//2, 80
    
    # Background rectangle
    padding = 15
    bg_x1 = max(0, position_x - padding)
    bg_y1 = max(0, position_y - text_height - padding)
    bg_x2 = min(width, position_x + text_width + padding)
    bg_y2 = min(height, position_y + padding)
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), 3)
    
    # Draw text with outline
    cv2.putText(frame, message, (position_x, position_y), font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(frame, message, (position_x, position_y), font, font_scale, (255, 255, 255), thickness)

# Open video (paths loaded from config.py)
video_capture = cv2.VideoCapture(config.INPUT_VIDEO)  # write the name of the video file here
output_file = config.OUTPUT_VIDEO  # extension the name and extension of the video file to be recorded

# Video features
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))

# Create the VideoWriter object to save the video file
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = video_capture.read()  # read video frame
    if not ret:
        break
    
    annotated_frame = frame.copy()
    results = model(annotated_frame, classes=TARGET_CLASSES, verbose=False)
    
    detections_info = []
    
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                bbox = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = bbox
                
                center_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names.get(class_id, "unknown")
                
                # Blur vehicle plates for all vehicles
                if class_name in ['car', 'motorcycle', 'bus', 'truck']:
                    blur_vehicle_plates_in_vehicle(annotated_frame, bbox)
                
                zone_name = get_vehicle_roi_zone(center_point)
                if zone_name is None or confidence <= CONFIDENCE_THRESHOLD:
                    continue
                
                distance = calculate_distance(bbox, class_id, zone_name)
                warning_distance = WARNING_DISTANCES.get(zone_name, 2.0)
                
                # Check if distance should be displayed based on ROI-specific thresholds
                if should_display_distance(distance, zone_name) and distance <= MAX_DISPLAY_DISTANCE:
                    color = get_distance_color(distance, zone_name)
                    draw_distance_label(annotated_frame, bbox, distance, color, zone_name)
                
                detections_info.append({
                    'distance': distance,
                    'warning_distance': warning_distance
                })
    
    # Display warning if any vehicle is too close
    if detections_info and any(d['distance'] < d['warning_distance'] for d in detections_info):
        draw_warning_message(annotated_frame, "WARNING: VEHICLE VERY CLOSE!")
    
    # Write the drawn frame to the video file to be saved
    output_video.write(annotated_frame)

    # Show video with vehicle distance measurement
    cv2.imshow('Vehicle Distance Measurement', annotated_frame)
    
    # Switch off video when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release all open windows
video_capture.release()
output_video.release()
cv2.destroyAllWindows()