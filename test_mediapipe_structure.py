#!/usr/bin/env python3
"""Cek struktur MediaPipe 0.10.31 secara mendalam."""

print("=" * 60)
print("  DEEP INSPECTION MEDIAPIPE 0.10.31")
print("=" * 60)

import mediapipe as mp
print(f"\n✅ MediaPipe version: {mp.__version__}")
print(f"📁 MediaPipe dir: {dir(mp)[:10]}")  # Show first 10 items

print("\n" + "=" * 60)
print("  CHECKING FOR 'pose' IN mp")
print("=" * 60)
if 'pose' in dir(mp):
    print("✅ mp.pose exists!")
    print(f"   Type: {type(mp.pose)}")
    print(f"   Has Pose class: {hasattr(mp.pose, 'Pose')}")
else:
    print("❌ mp.pose NOT exists")

print("\n" + "=" * 60)
print("  CHECKING FOR 'solutions' IN mp")
print("=" * 60)
if 'solutions' in dir(mp):
    print("✅ mp.solutions exists!")
else:
    print("❌ mp.solutions NOT exists")
    # Check for similar names
    print("\n   Checking for similar names:")
    for attr in dir(mp):
        if 'solution' in attr.lower() or 'pose' in attr.lower():
            print(f"   - {attr}")

print("\n" + "=" * 60)
print("  CHECKING ATTRIBUTES CONTAINING 'pose' or 'solution'")
print("=" * 60)
pose_attrs = [attr for attr in dir(mp) if 'pose' in attr.lower()]
solution_attrs = [attr for attr in dir(mp) if 'solution' in attr.lower()]

if pose_attrs:
    print(f"✅ Found {len(pose_attrs)} attributes with 'pose':")
    for attr in pose_attrs:
        print(f"   - {attr}")
else:
    print("❌ No attributes with 'pose'")

if solution_attrs:
    print(f"✅ Found {len(solution_attrs)} attributes with 'solution':")
    for attr in solution_attrs:
        print(f"   - {attr}")
else:
    print("❌ No attributes with 'solution'")

print("\n" + "=" * 60)
print("  TRYING DIRECT IMPORTS")
print("=" * 60)

# Try various direct imports
imports_to_try = [
    "from mediapipe import pose",
    "from mediapipe import Pose",
    "import mediapipe.pose",
    "from mediapipe.framework.formats import landmark_pb2",
]

for import_str in imports_to_try:
    try:
        exec(import_str)
        print(f"✅ {import_str}")
    except Exception as e:
        print(f"❌ {import_str} → {type(e).__name__}: {str(e)[:50]}")

print("\n" + "=" * 60)
print("  COMPLETE")
print("=" * 60)
