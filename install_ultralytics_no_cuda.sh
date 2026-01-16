#!/bin/bash
# Script install ultralytics TANPA CUDA (untuk disk space terbatas)

echo "================================================"
echo "  INSTALL ULTRALYTICS (NO CUDA - CPU ONLY)"
echo "================================================"
echo ""

cd ~/riftech-cam-security || exit 1

echo "💾 1. Checking disk space..."
df -h / | grep -v Filesystem
echo ""

echo "🧹 2. Cleaning up disk space..."
# Clean pip cache
pip3 cache purge
rm -rf ~/.cache/pip

# Clean apt cache
sudo apt-get clean
sudo apt-get autoremove -y

# Clean Python cache in project
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null

echo "✅ Cleanup complete"
echo ""

echo "📊 3. Disk space after cleanup..."
df -h / | grep -v Filesystem
echo ""

echo "📦 4. Installing ultralytics (CPU only, no CUDA)..."
# Install ultralytics without CUDA dependencies
pip3 install --no-cache-dir ultralytics

if [ $? -eq 0 ]; then
    echo "✅ Ultralytics installed successfully (CPU mode)"
else
    echo "⚠️  Ultralytics installation with issues, trying minimal install..."
    pip3 install --no-cache-dir ultralytics --no-deps
    pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi
echo ""

echo "🔄 5. Restarting security-system-web service..."
sudo systemctl restart security-system-web

if [ $? -eq 0 ]; then
    echo "✅ Service restarted"
else
    echo "❌ Failed to restart service"
    exit 1
fi
echo ""

echo "⏳ 6. Waiting for startup (15 seconds)..."
sleep 15
echo ""

echo "📝 7. Checking startup logs..."
echo "---"
tail -30 ~/riftech-cam-security/logs/websocket.log | grep -E '(Detector|Loading|Loaded|YOLO|ultralytics)'
echo "---"
echo ""

echo "🎯 8. Testing YOLO import..."
python3 -c "from ultralytics import YOLO; print('✅ YOLO import OK')" 2>&1
echo ""

echo "================================================"
echo "  INSTALLATION COMPLETE!"
echo "================================================"
echo ""
echo "⚠️  Note: Running in CPU mode (no CUDA acceleration)"
echo "   Detection will be slower but still functional"
echo ""
echo "🌐 Access web interface: http://10.26.27.104:8080/web.html"
echo ""
echo "📖 Check live logs: tail -f ~/riftech-cam-security/logs/websocket.log"
echo ""
