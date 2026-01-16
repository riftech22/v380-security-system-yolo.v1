#!/usr/bin/env python3
"""Debug split frame detection to see what's happening."""

import cv2
import numpy as np
from ultralytics import YOLO

print("=" * 70, flush=True)
print("Debugging Split Frame Detection", flush=True)
print("=" * 70, flush=True)
print()

# Get latest snapshot
import glob
snapshot_files = sorted(glob.glob('snapshots/*.jpg'), reverse=True)

if not snapshot_files:
    print("❌ No snapshots found!", flush=True)
    exit(1)

snapshot_path = snapshot_files[0]
print(f"Using snapshot: {snapshot_path}", flush=True)

frame = cv2.imread(snapshot_path)
if frame is None:
    print("❌ Failed to read snapshot!", flush=True)
    exit(1)

h, w = frame.shape[:2]
print(f"Frame size: {w}x{h}", flush=True)

# Split frame
mid_y = h // 2
top_frame = frame[:mid_y, :]
bottom_frame = frame[mid_y:, :]

print(f"\nTop frame: {top_frame.shape}", flush=True)
print(f"Bottom frame: {bottom_frame.shape}", flush=True)

# Load model
print(f"\nLoading YOLO model...", flush=True)
model = YOLO('yolov8m.pt')

# Detect in top frame
print(f"\n[1/2] Detecting in TOP frame...", flush=True)
results_top = model(top_frame, conf=0.25, classes=[0], verbose=False)
top_count = len(results_top[0].boxes) if results_top else 0
print(f"✅ Top camera persons detected: {top_count}", flush=True)

# Detect in bottom frame
print(f"\n[2/2] Detecting in BOTTOM frame...", flush=True)
results_bottom = model(bottom_frame, conf=0.15, classes=[0], verbose=False)
bottom_count = len(results_bottom[0].boxes) if results_bottom else 0
print(f"✅ Bottom camera persons detected: {bottom_count}", flush=True)

# Draw detections on full frame
output = frame.copy()

# Draw top camera detections
if results_top and results_top[0].boxes:
    for i, box in enumerate(results_top[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = float(box.conf[0])
        
        print(f"   Top person {i+1}: conf={conf:.2f}, bbox=({x1}, {y1}) -> ({x2}, {y2})", flush=True)
        
        # Draw in GREEN (top camera)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output, f"Top{conf:.0%}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Draw bottom camera detections
if results_bottom and results_bottom[0].boxes:
    for i, box in enumerate(results_bottom[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = float(box.conf[0])
        
        # Adjust coordinates to full frame
        y1_adj = y1 + mid_y
        y2_adj = y2 + mid_y
        
        print(f"   Bottom person {i+1}: conf={conf:.2f}, bbox=({x1}, {y1}) -> ({x2}, {y2}) -> adjusted: ({x1}, {y1_adj}) -> ({x2}, {y2_adj})", flush=True)
        
        # Draw in BLUE (bottom camera)
        cv2.rectangle(output, (x1, y1_adj), (x2, y2_adj), (255, 0, 0), 2)
        cv2.putText(output, f"Bot{conf:.0%}", (x1, y1_adj-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Draw split line
cv2.line(output, (0, mid_y), (w, mid_y), (0, 255, 255), 3)

# Save result
output_path = 'split_frame_debug.jpg'
cv2.imwrite(output_path, output)

print()
print("=" * 70, flush=True)
print("✅ Debug Complete!", flush=True)
print("=" * 70, flush=True)
print()
print(f"📊 Results:", flush=True)
print(f"   Top camera: {top_count} person(s)", flush=True)
print(f"   Bottom camera: {bottom_count} person(s)", flush=True)
print(f"   Total: {top_count + bottom_count} person(s)", flush=True)
print()
print(f"📸 Debug image saved: {output_path}", flush=True)
print()
print(f"📋 Check the image:", flush=True)
print(f"   - GREEN boxes = Top camera (0-{mid_y})", flush=True)
print(f"   - BLUE boxes = Bottom camera ({mid_y}-{h})", flush=True)
print(f"   - YELLOW line = Split boundary", flush=True)
print()
