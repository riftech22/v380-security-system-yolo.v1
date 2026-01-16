#!/bin/bash

###############################################################################
# Riftech Cam Security - Web Interface Startup Script
# This script starts the web server for headless operation
###############################################################################

# Add /usr/bin and /bin to PATH for systemd service
export PATH="/usr/bin:/bin:/usr/local/bin:$PATH"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "ERROR: Virtual environment not found!"
    echo "Please run installation script first: ./install_ubuntu_server.sh"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found in virtual environment!"
    exit 1
fi

# Check required Python packages
echo "Checking required Python packages..."
MISSING_PACKAGES=()

# Check each package
python -c "import cv2" 2>/dev/null || MISSING_PACKAGES+=("cv2")
python -c "import numpy" 2>/dev/null || MISSING_PACKAGES+=("numpy")
python -c "import ultralytics" 2>/dev/null || MISSING_PACKAGES+=("ultralytics")
python -c "import websockets" 2>/dev/null || MISSING_PACKAGES+=("websockets")
python -c "import torch" 2>/dev/null || MISSING_PACKAGES+=("torch")

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo "ERROR: Missing required packages: ${MISSING_PACKAGES[*]}"
    echo "Installing missing packages..."
    pip install -r requirements.txt
fi

echo "All required packages installed"

# Kill existing processes
echo "Cleaning up existing processes..."
pkill -f "python web_server.py" 2>/dev/null || true
pkill -f "python -m http.server" 2>/dev/null || true
sleep 2

echo "Cleanup complete"

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p recordings snapshots alerts trusted_faces logs fixed_images
echo "Directories created/verified"

# Start WebSocket server
echo "Starting WebSocket server on port 8765..."
python web_server.py > logs/websocket.log 2>&1 &
WS_PID=$!

# Wait for WebSocket server to start
sleep 3

# Check if WebSocket server is running
if ! kill -0 $WS_PID 2>/dev/null; then
    echo "ERROR: WebSocket server failed to start!"
    echo "Check logs: logs/websocket.log"
    cat logs/websocket.log
    exit 1
fi

echo "WebSocket server started (PID: $WS_PID)"

# Start HTTP server for web.html
echo "Starting HTTP server on port 8080..."
python3 -m http.server 8080 > logs/http.log 2>&1 &
HTTP_PID=$!

# Wait for HTTP server to start
sleep 2

# Check if HTTP server is running
if ! kill -0 $HTTP_PID 2>/dev/null; then
    echo "ERROR: HTTP server failed to start!"
    echo "Check logs: logs/http.log"
    cat logs/http.log
    exit 1
fi

echo "HTTP server started (PID: $HTTP_PID)"

# Get server IP
SERVER_IP=$(hostname -I | awk '{print $1}')

# Display success message
echo ""
echo "=========================================================="
echo "  SYSTEM STARTED SUCCESSFULLY!"
echo "=========================================================="
echo ""
echo "Riftech Cam Security Web Interface is now running!"
echo ""
echo "Access Information:"
echo "  - Web Interface: http://$SERVER_IP:8080/web.html"
echo "  - WebSocket:     ws://$SERVER_IP:8765"
echo ""
echo "Server PIDs:"
echo "  - WebSocket Server: $WS_PID"
echo "  - HTTP Server:      $HTTP_PID"
echo ""
echo "Log Files:"
echo "  - WebSocket: logs/websocket.log"
echo "  - HTTP:      logs/http.log"
echo ""
echo "To view logs in real-time:"
echo "  - WebSocket: tail -f logs/websocket.log"
echo "  - HTTP:      tail -f logs/http.log"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

# Trap for cleanup
trap cleanup SIGINT SIGTERM

cleanup() {
    echo ""
    echo "Stopping all servers..."
    
    if [ -n "$HTTP_PID" ] && kill -0 $HTTP_PID 2>/dev/null; then
        kill $HTTP_PID 2>/dev/null
        echo "HTTP server stopped"
    fi
    
    if [ -n "$WS_PID" ] && kill -0 $WS_PID 2>/dev/null; then
        kill $WS_PID 2>/dev/null
        echo "WebSocket server stopped"
    fi
    
    echo "All servers stopped. Goodbye!"
    exit 0
}

# Keep script running
while true; do
    # Check if servers are still running
    if ! kill -0 $WS_PID 2>/dev/null; then
        echo "ERROR: WebSocket server crashed! Check logs: logs/websocket.log"
        cleanup
    fi
    
    if ! kill -0 $HTTP_PID 2>/dev/null; then
        echo "ERROR: HTTP server crashed! Check logs: logs/http.log"
        cleanup
    fi
    
    sleep 5
done
