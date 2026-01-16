#!/usr/bin/env python3
"""Test akses pose lowercase dan cek cara inisialisasi."""

print("=" * 60)
print("  TEST POSE LOWERCASE & INITIALIZATION")
print("=" * 60)

import mediapipe as mp
print(f"\n✅ MediaPipe version: {mp.__version__}")

# Test 1: Akses pose lowercase
print("\n1. Testing mp.solutions.pose (lowercase)...")
if hasattr(mp, 'solutions'):
    if 'pose' in dir(mp.solutions):
        print("✅ 'pose' (lowercase) found!")
        print(f"   Type: {type(mp.solutions.pose)}")
        
        # Cek isi dari pose module
        print(f"   Dir: {dir(mp.solutions.pose)[:30]}")
    else:
        print("❌ 'pose' not found")

# Test 2: Cek Pose class di dalam pose module
print("\n2. Checking for Pose class in mp.solutions.pose...")
if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
    if 'Pose' in dir(mp.solutions.pose):
        print("✅ 'Pose' (uppercase) found in mp.solutions.pose!")
        print(f"   mp.solutions.pose.Pose = {mp.solutions.pose.Pose}")
    else:
        print("❌ 'Pose' not in mp.solutions.pose")
        print("   Available:", [x for x in dir(mp.solutions.pose) if not x.startswith('_')])

# Test 3: Coba inisialisasi Pose
print("\n3. Testing Pose initialization...")
if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
    try:
        pose_instance = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✅ Pose instance created successfully!")
        print(f"   Type: {type(pose_instance)}")
        print(f"   Has process method: {hasattr(pose_instance, 'process')}")
    except Exception as e:
        print(f"❌ Failed to create Pose instance")
        print(f"   Error: {type(e).__name__}: {str(e)[:80]}")

print("\n" + "=" * 60)
print("  COMPLETE")
print("=" * 60)
