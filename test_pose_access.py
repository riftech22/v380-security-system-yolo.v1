#!/usr/bin/env python3
"""Cek bagaimana mengakses Pose di MediaPipe 0.10.14."""

print("=" * 60)
print("  TEST POSE ACCESS IN MEDIAPIPE 0.10.14")
print("=" * 60)

import mediapipe as mp
print(f"\n✅ MediaPipe version: {mp.__version__}")

# Test 1: Cek mp.solutions structure
print("\n1. Checking mp.solutions structure...")
if hasattr(mp, 'solutions'):
    print("✅ mp.solutions exists")
    print(f"   Type: {type(mp.solutions)}")
    print(f"   Dir: {dir(mp.solutions)[:20]}")
else:
    print("❌ mp.solutions NOT exists")

# Test 2: Cek Pose di mp.solutions
print("\n2. Checking for Pose in mp.solutions...")
if hasattr(mp, 'solutions'):
    if 'Pose' in dir(mp.solutions):
        print("✅ 'Pose' found in mp.solutions.dir()")
        print(f"   mp.solutions.Pose = {mp.solutions.Pose}")
    else:
        print("❌ 'Pose' NOT in mp.solutions.dir()")
        print("   Available items:", [x for x in dir(mp.solutions) if not x.startswith('_')])

# Test 3: Coba import langsung
print("\n3. Testing direct imports...")
test_imports = [
    "from mediapipe.solutions import pose",
    "from mediapipe.solutions import Pose",
    "import mediapipe.solutions.pose",
    "from mediapipe import solutions as mp_solutions",
]

for import_str in test_imports:
    try:
        exec(import_str)
        print(f"✅ {import_str}")
    except Exception as e:
        print(f"❌ {import_str}")
        print(f"   Error: {type(e).__name__}: {str(e)[:60]}")

# Test 4: Cek jika Pose bisa diakses dengan cara lain
print("\n4. Testing alternative Pose access...")
if hasattr(mp, 'solutions'):
    try:
        # Coba akses sebagai attribute
        pose_class = getattr(mp.solutions, 'Pose', None)
        if pose_class:
            print(f"✅ mp.solutions.Pose accessible via getattr: {pose_class}")
        else:
            print("❌ Cannot access Pose via getattr")
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "=" * 60)
print("  COMPLETE")
print("=" * 60)
