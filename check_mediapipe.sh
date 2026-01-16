#!/bin/bash
echo "================================================"
echo "  MEDIAPIPE DETECTION CHECK"
echo "================================================"

echo ""
echo "1. === CHECK VENV ==="
if [ -d "venv" ]; then
    echo "✅ Venv exists"
    source venv/bin/activate
    echo "✅ Venv activated"
    echo "Python: $(python --version)"
else
    echo "❌ Venv not found"
    exit 1
fi

echo ""
echo "2. === CHECK MEDIAPIPE IN VENV ==="
if python -c "import mediapipe as mp; print('MediaPipe version:', mp.__version__)" 2>/dev/null; then
    echo "✅ MediaPipe installed in venv"
    python -c "import mediapipe as mp; print('✅ mp.solutions available:', hasattr(mp, 'solutions'))"
else
    echo "❌ MediaPipe NOT installed in venv"
    echo ""
    echo "   To install MediaPipe:"
    echo "   source venv/bin/activate"
    echo "   pip install mediapipe"
fi

echo ""
echo "3. === CHECK DETECTORS.PY ==="
if grep -q "MEDIAPIPE_AVAILABLE = True" detectors.py; then
    echo "✅ MEDIAPIPE_AVAILABLE = True in detectors.py"
else
    echo "❌ MEDIAPIPE_AVAILABLE = False in detectors.py"
fi

echo ""
echo "================================================"
echo "  CHECK COMPLETE"
echo "================================================"
