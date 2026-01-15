#!/bin/bash

echo "=== Starting Riftech Cam Security Web Interface ==="

# Kill existing processes
echo "Cleaning up existing processes..."
pkill -f "python web_server.py" 2>/dev/null
pkill -f "python main.py" 2>/dev/null
sleep 2

# Start Python web server
echo "Starting WebSocket server..."
cd ~/riftech-cam-security
source venv/bin/activate

# Check if websockets is installed
if ! python -c "import websockets" 2>/dev/null; then
    echo "Installing websockets..."
    pip install websockets
fi

# Run web server in background
python web_server.py &
WS_PID=$!
sleep 3

# Check if WebSocket server is running
if ! kill -0 $WS_PID 2>/dev/null; then
    echo "[ERROR] WebSocket server failed to start!"
    exit 1
fi

# Start HTTP server untuk web.html
echo "Starting HTTP server on port 8080..."
python3 -m http.server 8080 > http_server.log 2>&1 &
HTTP_PID=$!
sleep 2

# Check if HTTP server is running
if ! kill -0 $HTTP_PID 2>/dev/null; then
    echo "[ERROR] HTTP server failed to start!"
    exit 1
fi

echo ""
echo "=== Riftech Cam Security Web Interface Running Successfully! ==="
echo ""
echo "Access URL: http://$(hostname -I | awk '{print $1}'):8080/web.html"
echo "WebSocket: ws://$(hostname -I | awk '{print $1}'):8765"
echo ""
echo "Server PIDs:"
echo "  WebSocket Server: $WS_PID"
echo "  HTTP Server: $HTTP_PID"
echo ""
echo "Press Ctrl+C to stop all servers"

# Trap untuk cleanup
trap "
    echo ''
    echo 'Stopping all servers...'
    kill $HTTP_PID 2>/dev/null
    kill $WS_PID 2>/dev/null
    echo 'All servers stopped.'
    exit 0
" INT TERM

# Keep script running
wait
