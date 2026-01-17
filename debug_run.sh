#!/bin/bash

###############################################################################
# Debug Run Script - Run Server in Foreground Mode
###############################################################################

echo ""
echo "╔═════════════════════════════════════════════════════════════╗"
echo "║           🔍 DEBUG RUN - SECURITY SYSTEM                        ║"
echo "╚═════════════════════════════════════════════════════════════╝"
echo ""

# Stop all existing processes
echo "[1/4] Stopping existing processes..."
pkill -9 -f "web_server.py" 2>/dev/null
pkill -9 -f "http_server.py" 2>/dev/null
sleep 2
echo "      ✓ Done"
echo ""

# Check processes stopped
echo "[2/4] Verifying processes stopped..."
if ps aux | grep -E "(web_server|http_server)" | grep -v grep > /dev/null; then
    echo "      ⚠️  Warning: Some processes still running"
    ps aux | grep -E "(web_server|http_server)" | grep -v grep
else
    echo "      ✓ All processes stopped"
fi
echo ""

# Check config
echo "[3/4] Checking configuration..."
if [ -f "config.py" ]; then
    echo "      ✓ Config file found"
    
    # Check Telegram credentials
    if grep -q "TELEGRAM_BOT_TOKEN" config.py && grep -q "TELEGRAM_CHAT_ID" config.py; then
        echo "      ✓ Telegram credentials configured"
    else
        echo "      ⚠️  Warning: Telegram credentials missing"
    fi
    
    # Check camera source
    if grep -q "CAMERA_SOURCE" config.py; then
        CAMERA=$(grep "CAMERA_SOURCE" config.py | head -1 | cut -d'"' -f2)
        echo "      ✓ Camera source: $CAMERA"
    fi
else
    echo "      ❌ Config file not found!"
    exit 1
fi
echo ""

# Start server in foreground
echo "[4/4] Starting server in DEBUG mode..."
echo "      This will show ALL output in this terminal"
echo "      Press Ctrl+C to stop"
echo ""

# Activate virtual environment
echo "      Activating virtual environment..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "      ✓ Virtual environment activated"
    echo "      Python: $(which python3)"
else
    echo "      ⚠️  Virtual environment not found, using system Python"
fi
echo ""

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║           🚀 STARTING WEBSOCKET SERVER                         ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Run in foreground with all output visible
python3 web_server.py

# If server exits
echo ""
echo ""
echo "╔═════════════════════════════════════════════════════════════╗"
echo "║           ⚠️  SERVER STOPPED                                 ║"
echo "╚═════════════════════════════════════════════════════════════╝"
echo ""
echo "Check logs:"
echo "  tail -100 logs/websocket.log"
echo "  tail -100 logs/websocket_error.log"
echo ""
