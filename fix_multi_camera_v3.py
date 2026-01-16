#!/usr/bin/env python3
"""Test script untuk analisa multi-kamera split frame - Robust version."""

import cv2
import numpy as np
import subprocess
import os
import sys

def capture_frame_with_ffmpeg(rtsp_url, output_path="frame_temp.jpg", timeout=10):
    """Capture frame menggunakan FFmpeg CLI (bypass H.265 decoder issues)."""
    
    print(f"[1/4] Capturing frame with FFmpeg CLI...")
    print(f"     URL: {rtsp_url}")
    
    # FFmpeg command untuk capture 1 frame
    cmd = [
        'ffmpeg',
        '-rtsp_transport', 'tcp',
        '-i', rtsp_url,
        '-frames:v', '1',
        '-q:v', '2',  # Quality 2 (1-31, lower is better)
        '-timeout', str(timeout * 1000000),  # microseconds
        '-y',  # Overwrite output
        output_path
    ]
    
    print(f"     Command: {' '.join(cmd[:5])} ...")
    
    try:
        # Run FFmpeg
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout
        )
        
        print(f"     FFmpeg exit code: {result.returncode}")
        
        # Check jika file berhasil dibuat
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"     ✅ Output file created: {file_size} bytes")
            
            # Read frame dengan OpenCV
            frame = cv2.imread(output_path)
            
            if frame is not None:
                print(f"     ✅ Frame captured successfully: {frame.shape}")
                # Cleanup temp file
                os.remove(output_path)
                return frame
            else:
                print(f"     ❌ Failed to read frame from file")
                return None
        else:
            print(f"     ❌ FFmpeg failed to create output file")
            stderr_output = result.stderr.decode()
            if stderr_output:
                print(f"     FFmpeg stderr (last 500 chars): {stderr_output[-500:]}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"     ❌ FFmpeg timeout after {timeout} seconds")
        return None
    except Exception as e:
        print(f"     ❌ FFmpeg error: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_latest_snapshot():
    """Load latest snapshot from running service."""
    
    print(f"\n[2/4] Trying to load snapshot from service...")
    
    try:
        # Cek snapshots directory
        snapshot_dir = "snapshots"
        
        if not os.path.exists(snapshot_dir):
            print(f"     ❌ Snapshots directory not found: {snapshot_dir}")
            return None
        
        snapshots = [f for f in os.listdir(snapshot_dir) if f.endswith('.jpg')]
        
        if not snapshots:
            print(f"     ❌ No snapshots found in {snapshot_dir}")
            return None
        
        # Get latest snapshot
        latest_snapshot = sorted(snapshots)[-1]
        snapshot_path = os.path.join(snapshot_dir, latest_snapshot)
        
        print(f"     Found {len(snapshots)} snapshot(s)")
        print(f"     Latest: {latest_snapshot}")
        
        # Read frame
        frame = cv2.imread(snapshot_path)
        
        if frame is not None:
            file_size = os.path.getsize(snapshot_path)
            print(f"     ✅ Snapshot loaded: {frame.shape} ({file_size} bytes)")
            return frame
        else:
            print(f"     ❌ Failed to read snapshot")
            return None
            
    except Exception as e:
        print(f"     ❌ Error loading snapshot: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_multi_camera_split():
    """Analisa multi-kamera split frame."""
    
    print("=" * 70)
    print("Multi-Camera Split Analysis Tool v3")
    print("=" * 70)
    print()
    
    rtsp_url = r"rtsp://admin:Kuncong203@10.26.27.196:554/live/ch00_0"
    
    # Try capture dengan FFmpeg
    frame = capture_frame_with_ffmpeg(rtsp_url)
    
    # Fallback ke snapshot
    if frame is None:
        print(f"\n⚠️  FFmpeg failed, trying snapshot fallback...")
        frame = load_latest_snapshot()
        
        if frame is None:
            print(f"\n❌ Failed to get frame from RTSP and snapshot!")
            print(f"\n💡 Alternative: Take a snapshot from web interface first:")
            print(f"   1. Open http://10.26.27.104:8080/web.html")
            print(f"   2. Click snapshot button")
            print(f"   3. Run this script again")
            return
    
    h, w = frame.shape[:2]
    print(f"\n[3/4] Analyzing frame...")
    print(f"     Frame size: {w}x{h}")
    print(f"     Mid point: {w//2}x{h//2}")
    
    # Cek vertical split (paling umum untuk 2 camera)
    mid_y = h // 2
    top = frame[:mid_y, :]
    bottom = frame[mid_y:, :]
    
    print(f"     Top region: 0-{mid_y} (height: {mid_y})")
    print(f"     Bottom region: {mid_y}-{h} (height: {h-mid_y})")
    
    # Save sample images
    cv2.imwrite("multi_cam_full.jpg", frame)
    cv2.imwrite("multi_cam_top.jpg", top)
    cv2.imwrite("multi_cam_bottom.jpg", bottom)
    
    print(f"\n💾 Sample images saved:")
    print(f"     - multi_cam_full.jpg (full frame: {w}x{h})")
    print(f"     - multi_cam_top.jpg (top half: {w}x{mid_y})")
    print(f"     - multi_cam_bottom.jpg (bottom half: {w}x{h-mid_y})")
    
    # Test YOLO detection di masing-masing bagian
    print(f"\n[4/4] Testing YOLO detection...")
    
    try:
        from ultralytics import YOLO
        
        # Load model
        print(f"     Loading YOLO model...")
        model = YOLO('yolov8n.pt')
        print(f"     ✅ Model loaded successfully")
        
        # Detect top camera
        print(f"\n     📷 Detecting in TOP camera (0-{mid_y})...")
        results_top = model(top, conf=0.25, classes=[0], verbose=False)
        persons_top = len(results_top[0].boxes)
        print(f"     ✅ Top camera: {persons_top} person(s) detected")
        
        # Detect bottom camera
        print(f"\n     📷 Detecting in BOTTOM camera ({mid_y}-{h})...")
        results_bottom = model(bottom, conf=0.25, classes=[0], verbose=False)
        persons_bottom = len(results_bottom[0].boxes)
        print(f"     ✅ Bottom camera: {persons_bottom} person(s) detected")
        
        print(f"\n     📊 Total persons detected: {persons_top + persons_bottom}")
        
        # Draw detections
        if persons_top > 0:
            for box in results_top[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cv2.rectangle(top, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(top, f"{conf:.0%}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if persons_bottom > 0:
            for box in results_bottom[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cv2.rectangle(bottom, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(bottom, f"{conf:.0%}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save detection results
        cv2.imwrite("multi_cam_top_detected.jpg", top)
        cv2.imwrite("multi_cam_bottom_detected.jpg", bottom)
        
        print(f"\n     💾 Detection results saved:")
        print(f"         - multi_cam_top_detected.jpg")
        print(f"         - multi_cam_bottom_detected.jpg")
        
        # Analisa akurasi
        print(f"\n🔍 Accuracy Analysis:")
        print(f"     Top camera bounding boxes: {persons_top}")
        print(f"     Bottom camera bounding boxes: {persons_bottom}")
        
        if persons_top > 0 and persons_bottom > 0:
            print(f"     ✅ Both cameras detected!")
        elif persons_top > 0:
            print(f"     ⚠️  Only TOP camera detected (bottom might be empty)")
        elif persons_bottom > 0:
            print(f"     ⚠️  Only BOTTOM camera detected (top might be empty)")
        else:
            print(f"     ❌ No persons detected in either camera")
            print(f"     💡 Tip: Make sure someone is visible in camera view")
        
    except Exception as e:
        print(f"\n     ❌ YOLO test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        analyze_multi_camera_split()
        
        print()
        print("=" * 70)
        print("✅ Analysis Complete!")
        print("=" * 70)
        print()
        print("📋 Next steps:")
        print("   1. Check the saved images in current directory")
        print("   2. Send console output and screenshots to developer")
        print("   3. Wait for fix implementation")
        print()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
