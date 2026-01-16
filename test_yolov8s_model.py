#!/usr/bin/env python3
"""Test YOLOv8s model for better detection accuracy."""

print("=" * 70, flush=True)
print("Testing YOLOv8s Model for Better Detection", flush=True)
print("=" * 70, flush=True)
print()

try:
    from ultralytics import YOLO
    import cv2
    import numpy as np
    print("✅ Imports successful!", flush=True)
except ImportError as e:
    print(f"❌ Import error: {e}", flush=True)
    exit(1)

# Test both models
print("\n[1/3] Testing YOLOv8n (current model)...", flush=True)
try:
    model_n = YOLO('yolov8n.pt')
    print("✅ YOLOv8n loaded", flush=True)
except Exception as e:
    print(f"❌ YOLOv8n error: {e}", flush=True)
    model_n = None

print("\n[2/3] Testing YOLOv8s (larger model)...", flush=True)
try:
    model_s = YOLO('yolov8s.pt')
    print("✅ YOLOv8s loaded", flush=True)
except Exception as e:
    print(f"❌ YOLOv8s error: {e}", flush=True)
    model_s = None

# Get latest snapshot
print("\n[3/3] Testing detection on snapshot...", flush=True)

import os
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

# Test detection with both models
test_models = []
if model_n:
    test_models.append(('YOLOv8n', model_n))
if model_s:
    test_models.append(('YOLOv8s', model_s))

for model_name, model in test_models:
    print(f"\n{'='*70}", flush=True)
    print(f"Testing {model_name}", flush=True)
    print(f"{'='*70}", flush=True)
    
    # Test top camera
    print(f"\n📷 {model_name} - Top Camera:", flush=True)
    try:
        results_top = model(top_frame, conf=0.25, classes=[0], verbose=False)
        top_count = len(results_top[0].boxes) if results_top else 0
        print(f"   Persons detected: {top_count}", flush=True)
        
        if top_count > 0:
            for i, box in enumerate(results_top[0].boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                print(f"   Person {i+1}: conf={conf:.2f}, bbox=({int(x1)}, {int(y1)}) -> ({int(x2)}, {int(y2)})", flush=True)
    except Exception as e:
        print(f"   ❌ Error: {e}", flush=True)
    
    # Test bottom camera with different confidence thresholds
    thresholds = [0.25, 0.15, 0.10, 0.05]
    
    print(f"\n📷 {model_name} - Bottom Camera:", flush=True)
    
    for threshold in thresholds:
        try:
            results_bottom = model(bottom_frame, conf=threshold, classes=[0], verbose=False)
            bottom_count = len(results_bottom[0].boxes) if results_bottom else 0
            
            if bottom_count > 0:
                print(f"   ✅ Threshold {threshold}: {bottom_count} person(s) detected!", flush=True)
                for i, box in enumerate(results_bottom[0].boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    print(f"      Person {i+1}: conf={conf:.2f}, bbox=({int(x1)}, {int(y1)}) -> ({int(x2)}, {int(y2)})", flush=True)
                break  # Stop if found
            else:
                print(f"   ❌ Threshold {threshold}: No persons detected", flush=True)
        except Exception as e:
            print(f"   ❌ Error at threshold {threshold}: {e}", flush=True)

print()
print("=" * 70, flush=True)
print("✅ Test Complete!", flush=True)
print("=" * 70, flush=True)
print()
print("📋 Recommendation:", flush=True)
print("   - If YOLOv8s detects bottom camera better than YOLOv8n:", flush=True)
print("     → Change model to YOLOv8s in production", flush=True)
print("   - If both models fail to detect bottom camera:", flush=True)
print("     → Check lighting/angle of bottom camera", flush=True)
print("     → May need image preprocessing (brightness/contrast)", flush=True)
print()
