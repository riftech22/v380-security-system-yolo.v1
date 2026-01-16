#!/usr/bin/env python3
"""Test script untuk analisa multi-kamera split frame."""

import cv2
import numpy as np

def analyze_frame_split(frame_path):
    """Analisa bagaimana 2 kamera di-split."""
    
    cap = cv2.VideoCapture(frame_path)
    
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame!")
        return
    
    h, w = frame.shape[:2]
    print(f"Frame size: {w}x{h}")
    print(f"Mid point: {w//2}x{h//2}")
    
    # Cek horizontal split
    mid_x = w // 2
    left = frame[:, :mid_x]
    right = frame[:, mid_x:]
    
    # Cek vertical split
    mid_y = h // 2
    top = frame[:mid_y, :]
    bottom = frame[mid_y:, :]
    
    print("\n=== Analisa Split ===")
    
    # Save sample images
    cv2.imwrite("sample_full.jpg", frame)
    cv2.imwrite("sample_top.jpg", top)
    cv2.imwrite("sample_bottom.jpg", bottom)
    cv2.imwrite("sample_left.jpg", left)
    cv2.imwrite("sample_right.jpg", right)
    
    print("Sample images saved:")
    print("  - sample_full.jpg (full frame)")
    print("  - sample_top.jpg (top half)")
    print("  - sample_bottom.jpg (bottom half)")
    print("  - sample_left.jpg (left half)")
    print("  - sample_right.jpg (right half)")
    
    cap.release()

def test_from_rtsp():
    """Test split dari RTSP stream langsung."""
    
    rtsp_url = r"rtsp://admin:Kuncong203@10.26.27.196:554/live/ch00_0"
    
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print("Gagal connect ke RTSP!")
        return
    
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari RTSP!")
        return
    
    h, w = frame.shape[:2]
    print(f"RTSP Frame size: {w}x{h}")
    print(f"Mid point: {w//2}x{h//2}")
    
    # Cek vertical split (paling umum untuk 2 camera)
    mid_y = h // 2
    top = frame[:mid_y, :]
    bottom = frame[mid_y:, :]
    
    cv2.imwrite("rtsp_full.jpg", frame)
    cv2.imwrite("rtsp_top.jpg", top)
    cv2.imwrite("rtsp_bottom.jpg", bottom)
    
    print("\n=== RTSP Sample Images ===")
    print("Saved:")
    print("  - rtsp_full.jpg")
    print("  - rtsp_top.jpg")
    print("  - rtsp_bottom.jpg")
    
    # Test YOLO detection di masing-masing bagian
    print("\n=== Test YOLO Detection ===")
    
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        
        # Detect top camera
        results_top = model(top, conf=0.25, classes=[0], verbose=False)
        print(f"Top camera: {len(results_top[0].boxes)} persons detected")
        
        # Detect bottom camera
        results_bottom = model(bottom, conf=0.25, classes=[0], verbose=False)
        print(f"Bottom camera: {len(results_bottom[0].boxes)} persons detected")
        
        # Draw detections
        if len(results_top[0].boxes) > 0:
            for box in results_top[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(top, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if len(results_bottom[0].boxes) > 0:
            for box in results_bottom[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(bottom, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        cv2.imwrite("rtsp_top_detected.jpg", top)
        cv2.imwrite("rtsp_bottom_detected.jpg", bottom)
        print("\nDetection results saved:")
        print("  - rtsp_top_detected.jpg")
        print("  - rtsp_bottom_detected.jpg")
        
    except Exception as e:
        print(f"YOLO test error: {e}")
    
    cap.release()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test dari file gambar/video
        analyze_frame_split(sys.argv[1])
    else:
        # Test dari RTSP stream langsung
        print("Mencoba connect ke RTSP stream...")
        test_from_rtsp()
