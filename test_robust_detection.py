#!/usr/bin/env python3
"""Robust detection with brightness/contrast adjustment."""

import cv2
import numpy as np
from ultralytics import YOLO
import time

print("=" * 70, flush=True)
print("Robust Detection Test with Preprocessing", flush=True)
print("=" * 70, flush=True)
print()

# RTSP URL
RTSP_URL = "rtsp://admin:Kuncong203@10.26.27.196:554/"

print(f"Connecting to RTSP: {RTSP_URL}", flush=True)
print("Capturing frames with preprocessing...", flush=True)
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

# Capture and process frames
frames_processed = 0
best_person_count = 0
best_frame = None
best_results = None

while frames_processed < 20:  # Try up to 20 frames
    ret, frame = cap.read()
    
    if not ret or frame is None:
        print(f"⚠️  Failed to read frame {frames_processed+1}/20", flush=True)
        time.sleep(0.3)
        continue
    
    original_h, original_w = frame.shape[:2]
    
    # Resize to 1280x720
    frame_resized = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
    
    # Calculate brightness statistics
    brightness = np.mean(frame_resized)
    std_dev = np.std(frame_resized)
    
    # Apply brightness adjustment if overexposed
    frame_adjusted = frame_resized.copy()
    
    if brightness > 120:  # Overexposed
        # Darken the frame
        frame_adjusted = cv2.convertScaleAbs(frame_adjusted, alpha=0.7, beta=-30)
        print(f"Frame {frames_processed+1}: Brightness adjusted ({brightness:.1f} → overexposed)", flush=True)
    elif brightness < 50:  # Underexposed
        # Brighten the frame
        frame_adjusted = cv2.convertScaleAbs(frame_adjusted, alpha=1.3, beta=30)
        print(f"Frame {frames_processed+1}: Brightness adjusted ({brightness:.1f} → underexposed)", flush=True)
    
    # Apply CLAHE for better contrast
    lab = cv2.cvtColor(frame_adjusted, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    frame_adjusted = cv2.merge([l, a, b])
    frame_adjusted = cv2.cvtColor(frame_adjusted, cv2.COLOR_LAB2BGR)
    
    # Detect with VERY LOW confidence
    results = model(frame_adjusted, conf=0.15, classes=[0], verbose=False)
    person_count = len(results[0].boxes) if results and results[0].boxes else 0
    
    print(f"Frame {frames_processed+1}/20: {original_w}x{original_h} → 1280x720", flush=True)
    print(f"   Brightness: {brightness:.1f}, Std: {std_dev:.1f}", flush=True)
    print(f"   Persons detected: {person_count}", flush=True)
    
    if person_count > 0:
        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0])
            area = (x2 - x1) * (y2 - y1)
            print(f"      Person {i+1}: conf={conf:.2f}, bbox=({x1},{y1})->({x2},{y2}), area={area}", flush=True)
        
        # Keep best result
        if person_count > best_person_count:
            best_person_count = person_count
            best_frame = frame_adjusted.copy()
            best_results = results
            
            # Draw detections
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                cv2.rectangle(best_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(best_frame, f"{float(box.conf[0]):.2f}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            print(f"   ✅ NEW BEST: {best_person_count} persons!", flush=True)
    else:
        print(f"   No persons detected", flush=True)
    
    print()
    frames_processed += 1
    time.sleep(0.3)

cap.release()

print()
print("=" * 70, flush=True)
print("✅ Test Complete!", flush=True)
print("=" * 70, flush=True)
print()

if best_person_count > 0 and best_frame is not None:
    print(f"🎯 BEST RESULT: {best_person_count} persons detected!", flush=True)
    
    # Save best frame
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    best_path = f'robust_best_{timestamp}.jpg'
    cv2.imwrite(best_path, best_frame)
    print(f"📸 Best frame saved: {best_path}", flush=True)
    
    print()
    print("💡 SOLUTION WORKS!", flush=True)
    print("   1. RTSP stream: 2304x2592 → 1280x720 (RESIZED)", flush=True)
    print("   2. Brightness adjustment: Applied for over/underexposed frames", flush=True)
    print("   3. CLAHE contrast: Enhanced for better detection", flush=True)
    print("   4. YOLO detection: Working with confidence=0.15", flush=True)
    print()
    print("🔧 IMPLEMENTATION NEEDED:", flush=True)
    print("   Add preprocessing to main.py/web_server.py:", flush=True)
    print("   - cv2.resize() to 1280x720", flush=True)
    print("   - Brightness adjustment (alpha/beta)", flush=True)
    print("   - CLAHE for contrast enhancement", flush=True)
    print("   - Split frame detection logic", flush=True)
else:
    print(f"❌ No persons detected in {frames_processed} frames", flush=True)
    print()
    print("⚠️  POSSIBLE ISSUES:", flush=True)
    print("   1. No one standing in front of camera", flush=True)
    print("   2. Camera is pointing in wrong direction", flush=True)
    print("   3. Frame quality is too poor", flush=True)
    print("   4. Need even lower confidence threshold", flush=True)
    print()
    print("💡 RECOMMENDATIONS:", flush=True)
    print("   1. Stand in front of camera and run again", flush=True)
    print("   2. Check camera view via web interface", flush=True)
    print("   3. Try confidence=0.10", flush=True)
    print("   4. Check saved images in snapshots/", flush=True)
