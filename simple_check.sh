#!/bin/bash
echo "=== PYTHON SYNTAX CHECK ==="
cd ~/riftech-cam-security
python3 -m py_compile detectors.py 2>&1
echo "Exit code: $?"
echo ""

echo "=== CHECK YOLO PACKAGE ==="
pip3 list | grep -i ultralytics 2>&1
echo ""

echo "=== TEST YOLO IMPORT ==="
python3 -c "from ultralytics import YOLO; print('YOLO OK')" 2>&1
echo ""

echo "=== FULL STARTUP LOGS ==="
cat ~/riftech-cam-security/logs/websocket.log 2>&1
echo ""

echo "=== DONE ==="
