#!/bin/bash
# Reinstall MediaPipe dengan versi yang stabil

echo "============================================================"
echo "  REINSTALL MEDIAPIPE - STABLE VERSION"
echo "============================================================"

# Activate venv
source venv/bin/activate

echo ""
echo "1. Uninstall MediaPipe 0.10.31 (corrupt)..."
pip uninstall -y mediapipe

echo ""
echo "2. Install MediaPipe 0.10.14 (stable version)..."
pip install mediapipe==0.10.14

echo ""
echo "3. Verify installation..."
python3 -c "
import mediapipe as mp
print(f'✅ MediaPipe version: {mp.__version__}')
print(f'✅ Has solutions: {hasattr(mp, \"solutions\")}')
if hasattr(mp, 'solutions'):
    print(f'✅ Has Pose: {hasattr(mp.solutions, \"Pose\")}')
"

echo ""
echo "============================================================"
echo "  REINSTALLATION COMPLETE"
echo "============================================================"
