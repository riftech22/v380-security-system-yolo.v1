# 🔐 Riftech Cam Security System

**AI-Powered Security System with V380 Dual-Lens Camera Support**

<div align="center">

![Version](https://img.shields.io/badge/version-2.0.0-brightgreen.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Platform](https://img.shields.io/badge/platform-Linux-lightgrey.svg)
![Status](https://img.shields.io/badge/status-production-success.svg)

**Advanced AI-powered security camera system with real-time detection, web interface, and V380 dual-lens support**

</div>

---

## ✨ Features

### 🎯 AI Detection
- **Person Detection** - YOLOv8 powered human detection
- **Face Recognition** - Identify trusted persons
- **Motion Detection** - Advanced motion tracking
- **Skeleton Detection** - 33-point pose tracking
- **Heat Map** - Motion visualization
- **Night Vision** - Enhanced low-light mode

### 📹 V380 Dual-Lens Support
- **Split Frame Processing** - Automatically handle V380's vertical stacked dual-lens
- **FFmpeg Pipeline** - Real-time RTSP stream processing
- **Optimized YOLO** - Detection works perfectly on split frames
- **No Confusion** - AI correctly detects objects in both lenses

### 🌐 Web Interface
- **Real-time Streaming** - 30 FPS via WebSocket
- **Mobile Friendly** - Responsive design
- **Cyberpunk Theme** - Hacker-style UI
- **Multi-user Access** - Multiple clients
- **Zone Management** - Interactive zone drawing
- **All Controls** - ARM, RECORD, SNAPSHOT, MUTE

### 🔒 Security Features
- **Custom Security Zones** - Polygon-based monitoring
- **Breach Detection** - Real-time alerts
- **Trusted Persons** - Auto-disable for known faces
- **Telegram Bot** - Remote control & notifications
- **Audio Alerts** - Custom alarm sounds
- **Recording** - Save evidence automatically

### 🚀 Auto-Start (Systemd)
- **Auto-start on boot** - Services start automatically
- **Auto-restart** - Restart on crash (10 second delay)
- **Centralized logging** - Systemd journal + file logs
- **Easy management** - Simple systemctl commands

---

## 🚀 Quick Start

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3 python3-pip python3-venv python3-dev
sudo apt install -y ffmpeg libopencv-dev libavcodec-dev libavformat-dev libswscale-dev
```

### Installation

```bash
# 1. Clone repository
git clone https://github.com/riftech22/riftech-cam-security.git
cd riftech-cam-security

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure V380 camera
cp config.example.py config.py
nano config.py  # Edit RTSP_URL and other settings
```

### Start System

**Option 1: Manual Start**
```bash
# Start both servers
./start_both_servers.sh

# Access web interface
# http://YOUR_IP:8080/web.html
```

**Option 2: Auto-Start (Recommended)**
```bash
# Install systemd services
sudo ./install_services.sh

# Services will auto-start on boot!
```

---

## 📹 V380 Dual-Lens Setup

### What is V380 Dual-Lens?

V380 dual-lens cameras have **two sensors** stacked vertically in a single frame:

```
┌─────────────────────────────────────┐
│         LENS ATAS (Fixed)           │  ← Wide angle monitoring
│         0 to 360px                 │
├─────────────────────────────────────┤
│         LENS BAWAH (PTZ)            │  ← Pan-tilt-zoom
│        360 to 720px                │
└─────────────────────────────────────┘
```

### How It Works

The system uses **FFmpeg pipeline** to:
1. Capture RTSP stream from V380 camera
2. **Automatically split** the frame into two parts
3. **Process each part separately** with YOLO
4. **Merge detections** from both lenses
5. **Stream to web interface** with proper visualization

### Configure V380 Camera

Edit `config.py`:

```python
# V380 RTSP URL
RTSP_URL = "rtsp://admin:password@IP:554/h264/ch1/main/av_stream"

# Model settings
MODEL_NAME = 'yolov8s.pt'  # Use small model for better accuracy
CONFIDENCE = 0.25  # 25% confidence threshold

# V380 mode (IMPORTANT!)
V380_MODE = True  # Enable V380 split frame processing
```

### Example RTSP URL for V380

```python
# Format: rtsp://USERNAME:PASSWORD@CAMERA_IP:PORT/PATH
RTSP_URL = "rtsp://admin:Kuncong203@10.26.27.196:554/h264/ch1/main/av_stream"
```

---

## 📁 Project Structure

```
riftech-cam-security/
│
├── 📄 README.md                      # Main documentation
├── 📄 SERVICES_README.md            # Systemd services documentation
├── 📄 requirements.txt              # Python dependencies
│
├── 🐍 web_server.py                # WebSocket server + V380 pipeline
├── 🐍 http_server.py               # HTTP server for web interface
├── 🐍 v380_ffmpeg_pipeline.py      # V380 FFmpeg processing
├── 🐍 config.py                    # Configuration
├── 🐍 detectors.py                 # Detection modules
├── 🐍 database.py                  # Database manager
├── 🐍 telegram_bot.py              # Telegram integration
├── 🐍 audio.py                     # Audio system
├── 🐍 utils.py                     # Utilities
│
├── 🌐 web.html                     # Single-file web interface
│
├── 🚀 start_both_servers.sh         # Start HTTP + WebSocket
├── 🛑 stop_both_servers.sh          # Stop both servers
├── ⚙️  install_services.sh          # Install systemd services
│
├── 📜 security-system-v380.service # WebSocket systemd service
├── 📜 http-server.service          # HTTP systemd service
│
├── 📂 recordings/                  # Saved videos
├── 📂 snapshots/                   # Screenshots
├── 📂 alerts/                      # Alert photos
├── 📂 trusted_faces/               # Trusted person photos
├── 📂 logs/                        # System logs
│
├── 📂 audio/                       # Audio files
│   └── alarm.wav
│
└── 🐁 venv/                       # Virtual environment
```

---

## 🎮 Web Interface Guide

### Access

```bash
# Start servers
./start_both_servers.sh

# Open in browser
http://YOUR_IP:8080/web.html
```

### Features

#### Control Panel
- **🔒 ARM SYSTEM** - Activate/deactivate monitoring
- **⏺ RECORD** - Start/stop video recording
- **📸 SNAPSHOT** - Capture screenshot
- **🔇 MUTE** - Toggle alarm audio
- **🎯 CONFIDENCE** - Adjust detection sensitivity (15-50%)
- **🤖 MODEL** - Switch YOLO models (Nano/Small/Medium)
- **🦴 SKELETON** - Toggle pose tracking
- **👤 FACE** - Toggle face recognition
- **📡 MOTION** - Toggle motion boxes
- **🔥 HEAT MAP** - Toggle motion heatmap
- **🌙 NIGHT VISION** - Toggle night mode
- **➕ NEW ZONE** - Create security zone
- **✏️ DRAW** - Draw zone polygons
- **🗑️ CLEAR** - Remove all zones
- **🔄 RELOAD FACES** - Update trusted faces

### Creating Security Zones

1. Click **NEW ZONE** button
2. Click **DRAW** to enter drawing mode
3. Click 3+ points on video feed to create polygon
4. Click **DRAW** again to complete zone
5. Click **ARM SYSTEM** to activate monitoring

### Face Recognition Setup

```bash
# Add photos to trusted_faces folder
cp photo1.jpg trusted_faces/
cp photo2.jpg trusted_faces/

# Reload from web interface
# Click "RELOAD FACES" button
```

---

## 🚀 Systemd Services (Auto-Start)

### Install Services

```bash
# Run installation script
sudo ./install_services.sh
```

This will:
1. ✅ Check all requirements
2. ✅ Create logs directory
3. ✅ Copy service files to `/etc/systemd/system/`
4. ✅ Enable services for auto-start on boot
5. ✅ Start services
6. ✅ Check status

### Service Management

```bash
# Check status
sudo systemctl status security-system-v380
sudo systemctl status http-server

# Start/Stop/Restart
sudo systemctl restart security-system-v380
sudo systemctl restart http-server

# View logs
sudo journalctl -u security-system-v380 -f
sudo journalctl -u http-server -f
```

### Service Files

**security-system-v380.service** - WebSocket server with V380 support
- **Port**: 8765 (WebSocket)
- **Mode**: V380 FFmpeg
- **Auto-restart**: Yes (10 second delay)

**http-server.service** - HTTP server for web interface
- **Port**: 8080 (HTTP)
- **Auto-restart**: Yes (10 second delay)

For detailed documentation, see `SERVICES_README.md`.

---

## 🔧 Configuration

### Detection Settings

**Confidence Levels:**
- **15%** - High sensitivity (may have false positives)
- **25%** - Medium (recommended for most scenarios)
- **50%** - Low sensitivity (fewer false alarms)

**YOLO Models:**
- **Nano (yolov8n.pt)** - Fastest, lightweight (~6MB)
- **Small (yolov8s.pt)** - Balanced speed/accuracy (~22MB) **[Recommended for V380]**
- **Medium (yolov8m.pt)** - Most accurate (~50MB)

### Performance Tuning

**For Low-End Systems (512MB RAM):**
```python
# Use Nano model
MODEL_NAME = 'yolov8n.pt'

# Increase confidence
CONFIDENCE = 0.40  # 40%

# Disable features
ENABLE_SKELETON = False
ENABLE_HEATMAP = False
```

**For High-End Systems (4GB+ RAM):**
```python
# Use Small or Medium model
MODEL_NAME = 'yolov8s.pt'  # or 'yolov8m.pt'

# Lower confidence
CONFIDENCE = 0.20  # 20%

# Enable features
ENABLE_SKELETON = True
ENABLE_HEATMAP = True
```

---

## 📊 System Requirements

### Minimum Requirements
- **OS**: Ubuntu 20.04+, Debian 11+
- **CPU**: Dual-core 2.0 GHz
- **RAM**: 2 GB (4 GB recommended)
- **Storage**: 500 MB for application, 5 GB+ for recordings
- **Camera**: V380 dual-lens or IP camera with RTSP
- **Python**: 3.8 or higher
- **FFmpeg**: Required for video processing

### Recommended Requirements
- **CPU**: Quad-core 3.0 GHz+
- **RAM**: 8 GB+
- **Storage**: 20 GB+ SSD
- **Network**: 100 Mbps+ for streaming

---

## 🔍 Troubleshooting

### Camera Not Connecting

```bash
# Test RTSP connection manually
ffplay rtsp://admin:password@CAMERA_IP:554/h264/ch1/main/av_stream

# Check network connectivity
ping CAMERA_IP

# Check firewall
sudo ufw status
sudo ufw allow 8765/tcp
sudo ufw allow 8080/tcp
```

### High CPU Usage

**Solutions:**
1. Use Nano model instead of Small/Medium
2. Increase confidence to 30-40%
3. Disable skeleton and heatmap
4. Reduce FPS in config

### WebSocket Connection Failed

```bash
# Check if port is listening
netstat -tlnp | grep 8765

# Check server logs
tail -f logs/websocket.log

# Restart services
sudo systemctl restart security-system-v380
```

### V380 Detection Issues

**Problem**: AI detects objects incorrectly on split frames

**Solution**: 
1. Ensure `V380_MODE = True` in config.py
2. Check RTSP URL is correct
3. Verify camera is V380 dual-lens
4. Restart servers

### Services Not Starting on Boot

```bash
# Check if enabled
sudo systemctl is-enabled security-system-v380
sudo systemctl is-enabled http-server

# Enable if disabled
sudo systemctl enable security-system-v380
sudo systemctl enable http-server

# Check network dependency
sudo systemctl status network-online.target
```

---

## 📞 Support & Documentation

- **GitHub Issues**: https://github.com/riftech22/riftech-cam-security/issues
- **Systemd Services Guide**: See `SERVICES_README.md`
- **Web Interface Guide**: See `WEB_INTERFACE_README.md`

---

## 🔄 Updates

```bash
# Pull latest changes
git pull origin main

# Update dependencies
source venv/bin/activate
pip install -r requirements.txt --upgrade

# Restart services
sudo systemctl restart security-system-v380 http-server
```

---

## 📚 Additional Resources

- **YOLOv8 Documentation**: https://docs.ultralytics.com/
- **OpenCV Documentation**: https://docs.opencv.org/
- **FFmpeg Documentation**: https://ffmpeg.org/documentation.html
- **MediaPipe**: https://mediapipe.dev/

---

## 🙏 Acknowledgments

- **YOLOv8** by Ultralytics
- **OpenCV** by OpenCV Team
- **MediaPipe** by Google
- **FFmpeg** by FFmpeg Team

---

## 📄 License

This project is open source and available under the MIT License.

---

<div align="center">

## 🎉 Ready to Secure Your Space?

**Clone, Install, and Start Monitoring Today!**

```bash
git clone https://github.com/riftech22/riftech-cam-security.git
cd riftech-cam-security
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./start_both_servers.sh
```

**Made with ❤️ by Riftech**

</div>
