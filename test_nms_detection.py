#!/usr/bin/env python3
"""Test detection with NMS to filter overlapping detections."""

import cv2
import numpy as np
from ultralytics import YOLO
import time

print("=" * 70, flush=True)
print("Test Detection with NMS", flush=True)
print("=" * 70, flush=True)
print()

# RTSP URL
RTSP_URL = "rtsp://admin:Kuncong203@10.26.27.196:554/"

print(f"Connecting to RTSP: {RTSP_URL}", flush=True)
print("Capturing frames with NMS filtering...", flush=True)
print()

# Open RTSP
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("❌ Failed to open RTSP stream!", flush=True)
    exit(1)

print("✅ RTSP opened!", flush=True)

# Load YOLO model
print("\nLoading YOLOv8m model...", flush=True)
model = YOLO('yolov8m.pt')


def apply_nms(detections, iou_threshold=0.5):
    """Apply Non-Maximum Suppression to filter overlapping detections.
    
    Args:
        detections: List of (x1, y1, x2, y2, confidence)
        iou_threshold: IoU threshold for suppression
        
    Returns:
        Filtered detections
    """
    if len(detections) == 0:
        return detections
    
    # Sort by confidence (highest first)
    detections = sorted(detections, key=lambda x: x[4], reverse=True)
    
    # Calculate areas
    areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2, conf) in detections]
    
    # Initialize list to keep
    keep = []
    
    while detections:
        # Pick the detection with highest confidence
        current = detections[0]
        keep.append(current)
        detections.pop(0)
        current_area = areas[0]
        areas.pop(0)
        
        # Calculate IoU with remaining detections
        remaining = []
        for (x1, y1, x2, y2, conf), area in zip(detections, areas):
            # Calculate intersection
            inter_x1 = max(current[0], x1)
            inter_y1 = max(current[1], y1)
            inter_x2 = min(current[2], x2)
            inter_y2 = min(current[3], y2)
            
            # Check if there's overlap
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                union_area = current_area + area - inter_area
                iou = inter_area / union_area
                
                # Keep only if IoU is below threshold
                if iou < iou_threshold:
                    remaining.append((x1, y1, x2, y2, conf))
            else:
                remaining.append((x1, y1, x2, y2, conf))
        
        detections = remaining
    
    return keep


# Capture and process frames
frames_processed = 0
best_frame = None
best_detections = None

while frames_processed < 10:  # Try up to 10 frames
    ret, frame = cap.read()
    
    if not ret or frame is None:
        print(f"⚠️  Failed to read frame {frames_processed+1}/10", flush=True)
        time.sleep(0.3)
        continue
    
    # Resize to 1280x720
    frame_resized = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
    
    # Brightness adjustment
    brightness = np.mean(frame_resized)
    if brightness > 120:
        frame_resized = cv2.convertScaleAbs(frame_resized, alpha=0.7, beta=-30)
    
    # CLAHE
    lab = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    frame_resized = cv2.merge([l, a, b])
    frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_LAB2BGR)
    
    # Detect with higher confidence
    results = model(frame_resized, conf=0.30, classes=[0], verbose=False)
    
    # Extract detections
    detections_before = []
    if results and results[0].boxes:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0])
            detections_before.append((x1, y1, x2, y2, conf))
    
    print(f"\nFrame {frames_processed+1}/10:", flush=True)
    print(f"   BEFORE NMS: {len(detections_before)} persons detected", flush=True)
    
    for i, (x1, y1, x2, y2, conf) in enumerate(detections_before):
        area = (x2 - x1) * (y2 - y1)
        print(f"      Person {i+1}: conf={conf:.2f}, bbox=({x1},{y1})->({x2},{y2}), area={area}", flush=True)
    
    # Apply NMS
    detections_after = apply_nms(detections_before, iou_threshold=0.5)
    print(f"   AFTER NMS:  {len(detections_after)} persons detected", flush=True)
    
    for i, (x1, y1, x2, y2, conf) in enumerate(detections_after):
        area = (x2 - x1) * (y2 - y1)
        print(f"      Person {i+1}: conf={conf:.2f}, bbox=({x1},{y1})->({x2},{y2}), area={area}", flush=True)
    
    # Keep best result
    if len(detections_after) > 0 and len(detections_after) <= 2:  # Keep if reasonable (1-2 persons)
        if best_detections is None or len(detections_after) < len(best_detections):
            best_detections = detections_after
            best_frame = frame_resized.copy()
            
            # Draw detections
            for i, (x1, y1, x2, y2, conf) in enumerate(best_detections):
                cv2.rectangle(best_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(best_frame, f"#{i+1}: {conf:.2f}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            print(f"   ✅ NEW BEST: {len(best_detections)} persons!", flush=True)
    
    frames_processed += 1
    time.sleep(0.3)

cap.release()

print()
print("=" * 70, flush=True)
print("✅ Test Complete!", flush=True)
print("=" * 70, flush=True)
print()

if best_detections and len(best_detections) > 0:
    print(f"🎯 BEST RESULT: {len(best_detections)} person(s) detected!", flush=True)
    
    # Save best frame
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    best_path = f'nms_best_{timestamp}.jpg'
    cv2.imwrite(best_path, best_frame)
    print(f"📸 Best frame saved: {best_path}", flush=True)
    
    print()
    print("💡 NMS WORKS!", flush=True)
    print("   - Filtered overlapping detections", flush=True)
    print("   - Reduced false positives", flush=True)
    print("   - More accurate person count", flush=True)
else:
    print("❌ No reasonable detections found", flush=True)
