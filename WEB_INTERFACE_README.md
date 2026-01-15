# 🖥️ Riftech Cam Security - Web Interface

Web interface cyberpunk-styled lengkap untuk AI-Powered Security System dengan semua kontrol yang ada di GUI PyQt6.

## ✨ Fitur

### 📹 Video Monitoring
- Real-time video streaming (30 FPS)
- WebSocket untuk low-latency
- Responsive design untuk semua device
- Canvas overlay untuk zone drawing

### ⚡ Controls
- **ARM/DISARM** - Aktifkan/matikan system
- **RECORD** - Mulai/hentikan recording
- **SNAPSHOT** - Ambil screenshot
- **MUTE** - Matikan/nyalakan alarm

### 🎯 Detection Settings
- **Confidence Level** - 15% sampai 50%
- **YOLO Model** - Nano, Small, Medium
- **Skeleton Detection** - Toggle 33-point pose tracking
- **Face Recognition** - Toggle trusted person detection
- **Motion Detection** - Toggle motion boxes
- **Heat Map** - Toggle motion visualization
- **Night Vision** - Toggle green night vision mode

### 🎯 Zone Management
- **NEW ZONE** - Buat zone baru
- **DRAW** - Mode drawing untuk tambah titik zone
- **CLEAR ALL ZONES** - Hapus semua zones
- Klik pada video feed untuk menambah titik zone
- Minimum 3 titik untuk zone

### 👤 Trusted Faces
- Reload trusted faces dari folder
- Display jumlah faces yang loaded
- Auto-disable alarm untuk trusted persons

### 📊 Statistics
- Real-time FPS counter
- Connected clients count
- Breach duration timer
- Person count
- Alert count

### 📋 Activity Log
- Real-time activity log
- Color-coded messages (success, warning, error)
- Auto-scroll ke latest
- Max 100 entries

---

## 📦 Prerequisites

### Ubuntu Server 22.04

```bash
# Update system
sudo apt update

# Install Python dan pip
sudo apt install -y python3 python3-pip python3-venv

# Install OpenCV dan dependencies
sudo apt install -y \
    python3-dev \
    libopencv-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    cmake \
    g++

# Install system dependencies untuk dlib
sudo apt install -y \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    libboost-python-dev \
    libboost-all-dev
```

### Clone Repository

```bash
cd ~
git clone https://github.com/riftech22/riftech-cam-security.git
cd riftech-cam-security
```

### Setup Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

# Install websockets untuk web interface
pip install websockets

# Install PyTorch (CPU version untuk server)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install ultralytics (YOLOv8)
pip install ultralytics
```

---

## 🚀 Quick Start

### Manual Start

```bash
cd ~/riftech-cam-security
chmod +x start_web.sh
./start_web.sh
```

### Auto-Start on Reboot (Systemd)

#### 1. Edit Service File

```bash
nano security-system-web.service
```

Ganti `YOUR_USERNAME` dengan username Anda:

```ini
[Unit]
Description=Riftech Cam Security Web Interface
After=network.target

[Service]
Type=simple
User=riftech
WorkingDirectory=/home/riftech/riftech-cam-security
Environment="PATH=/home/riftech/riftech-cam-security/venv/bin"
ExecStart=/home/riftech/riftech-cam-security/start_web.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### 2. Install Service

```bash
# Copy ke systemd
sudo cp security-system-web.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable auto-start on boot
sudo systemctl enable security-system-web

# Start service
sudo systemctl start security-system-web

# Check status
sudo systemctl status security-system-web
```

#### 3. Service Management

```bash
# Start service
sudo systemctl start security-system-web

# Stop service
sudo systemctl stop security-system-web

# Restart service
sudo systemctl restart security-system-web

# Check logs
sudo journalctl -u security-system-web -f
```

---

## 🌐 Access Web Interface

### Local Network

Buka browser dan akses:
```
http://YOUR_SERVER_IP:8080
```

Contoh:
- `http://192.168.1.100:8080`
- `http://10.0.0.5:8080`

Cara cek IP server:
```bash
hostname -I
```

### Remote Access (Public Server)

Jika server punya public IP:
```
http://YOUR_PUBLIC_IP:8080
```

**Security Note**: Untuk production, setup firewall dan HTTPS!

### Firewall Configuration

```bash
# Allow port 8080 (HTTP untuk HTML)
sudo ufw allow 8080/tcp

# Allow port 8765 (WebSocket untuk video stream)
sudo ufw allow 8765/tcp

# Enable firewall
sudo ufw enable

# Check status
sudo ufw status
```

---

## 🎮 Penggunaan Web Interface

### 1. Connection Status

Setelah buka web interface, Anda akan melihat:
- **Connected** (hijau) - WebSocket connected
- **Disconnected** (merah) - Reconnecting...

Jika disconnected, tunggu beberapa detik untuk auto-reconnect.

### 2. Video Feed

Video feed akan menampilkan:
- Real-time camera feed
- Detection boxes (person, motion)
- Zone overlays
- Recording indicator
- Armed status

### 3. Arm System

Klik **ARM SYSTEM** untuk mengaktifkan monitoring:
- Zone akan berubah warna (cyan → red jika breach)
- Deteksi aktif
- Alarm akan berbunyi jika breach

Klik lagi untuk DISARM.

### 4. Create Security Zones

1. Klik **NEW ZONE**
2. Klik **DRAW** untuk masuk mode drawing
3. Klik pada video feed untuk tambah titik (minimum 3)
4. Zone akan otomatis complete setelah 3+ titik
5. Klik **DRAW** lagi untuk keluar dari drawing mode

Tips:
- Gambar polygon di sekitar area yang mau diamankan
- Zone akan berkedip (cyan) saat armed
- Zone akan merah saat breach detected

### 5. Take Snapshot

Klik **SNAPSHOT** untuk:
- Ambil screenshot saat ini
- Simpan ke `snapshots/` folder
- Tampilkan di modal popup
- Download dari popup

### 6. Record Video

Klik **RECORD** untuk:
- Mulai recording ke `recordings/` folder
- Indicator merah akan muncul di video
- Klik lagi untuk stop

### 7. Detection Settings

#### Confidence Level
- **15%** - High sensitivity (bisa banyak false positives)
- **25%** - Medium (default, balanced)
- **50%** - Low sensitivity (kurang false positives)

#### YOLO Model
- **Nano (Fast)** - Paling cepat, kurang akurat
- **Small (Balanced)** - Balance speed/accuracy
- **Medium (Accurate)** - Paling akurat, lebih lambat

#### Toggle Features
- **Skeleton Detection** - Tampilkan 33-point pose
- **Face Recognition** - Detect dan recognize trusted faces
- **Motion Detection** - Tampilkan motion boxes
- **Heat Map** - Overlay motion heatmap
- **Night Vision** - Mode malam (green tint)

### 8. Trusted Faces

Klik **RELOAD FACES** untuk:
- Reload faces dari `trusted_faces/` folder
- Update jumlah faces yang loaded
- Auto-disable alarm untuk trusted persons

Menambah trusted faces:
```bash
# Copy foto ke folder
cp your_photo.jpg ~/riftech-cam-security/trusted_faces/

# Reload dari web interface
```

### 9. Activity Log

Activity log menampilkan:
- Timestamp
- Action yang dilakukan
- Status (success, warning, error)

Auto-scroll ke latest. Max 100 entries.

---

## 🔧 Troubleshooting

### Camera Not Detected

```bash
# Check available cameras
ls -l /dev/video*

# Test camera
ffplay /dev/video0

# Add user to video group
sudo usermod -a -G video $USER

# Re-login
newgrp video
```

### WebSocket Connection Failed

```bash
# Check if port 8765 listening
netstat -tlnp | grep 8765

# Check firewall
sudo ufw status

# Allow port
sudo ufw allow 8765/tcp
```

### HTTP Not Accessible

```bash
# Check if server running
ps aux | grep python

# Check port 8080
netstat -tlnp | grep 8080

# Check firewall
sudo ufw status

# Allow port
sudo ufw allow 8080/tcp
```

### High CPU Usage

```bash
# Check CPU
htop

# Solution: Gunakan model Nano (lebih ringan)
# Atau kurangi confidence (lebih sedikit detections)
```

### Out of Memory

```bash
# Check memory
free -h

# Add swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Set swappiness
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
```

### Detection Not Working

```bash
# Check if models downloaded
ls ~/.cache/ultralytics/

# Manually download model
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Check detection modules
source venv/bin/activate
python -c "from detectors import PersonDetector; print('OK')"
```

---

## 📊 Performance Tips

### Untuk Server Kecil (512MB RAM)

- Gunakan YOLOv8 Nano model
- Matikan skeleton detection
- Matikan heat map
- Confidence: 30-50%

### Untuk Server Medium (1GB RAM)

- Gunakan YOLOv8 Small model
- Skeleton: Off
- Heat map: Off
- Confidence: 25-30%

### Untuk Server Besar (2GB+ RAM)

- Gunakan YOLOv8 Medium model
- Skeleton: On
- Heat map: On
- Confidence: 15-25%

### Untuk GPU Available

Install PyTorch dengan CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🔒 Security Recommendations

### Production Setup

1. **Setup Firewall**
```bash
sudo ufw allow 8080/tcp
sudo ufw allow 8765/tcp
sudo ufw enable
```

2. **Setup HTTPS dengan Nginx**

Install Nginx:
```bash
sudo apt install -y nginx
```

Config Nginx (`/etc/nginx/sites-available/security-system`):
```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    location /ws/ {
        proxy_pass http://localhost:8765;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

Enable config:
```bash
sudo ln -s /etc/nginx/sites-available/security-system /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

3. **Setup Authentication**

Tambahkan password protection dengan htpasswd:
```bash
sudo apt install -y apache2-utils
sudo htpasswd -c /etc/nginx/.htpasswd admin

# Edit nginx config
# Add:
# auth_basic "Restricted";
# auth_basic_user_file /etc/nginx/.htpasswd;
```

---

## 📁 File Structure

```
riftech-cam-security/
├── web_server.py              # WebSocket server backend
├── web.html                   # Web interface frontend
├── start_web.sh               # Start script
├── security-system-web.service # Systemd service
├── gui.py                     # Original GUI (PyQt6)
├── detectors.py                # Detection modules
├── config.py                  # Configuration
├── database.py                # Database manager
├── telegram_bot.py            # Telegram bot
├── audio.py                   # Audio system
├── utils.py                   # Zone utilities
├── venv/                      # Virtual environment
├── recordings/               # Saved videos
├── snapshots/                # Saved snapshots
├── alerts/                   # Alert photos
├── trusted_faces/            # Trusted person photos
└── logs/                     # System logs
```

---

## 🆚 Perbedaan dengan GUI PyQt6

### Web Interface
✅ Ringan (RAM: ~50-100MB)
✅ Cepat (30+ FPS)
✅ Low bandwidth (~1 Mbps)
✅ Mobile friendly
✅ Multi-user access
✅ No X11 dependencies
✅ Easy deployment
❌ Harus implement zone drawing manual
❌ Tidak ada native drag-and-drop
❌ Butuh HTML/JS coding untuk features baru

### GUI PyQt6
✅ GUI penuh dengan native controls
✅ Drag-and-drop native
✅ Zone drawing built-in
✅ Semua features out-of-the-box
❌ Berat (RAM: ~150-200MB dengan Xvfb)
❌ Lambat (10-20 FPS dengan VNC)
❌ High bandwidth (~2-5 Mbps)
❌ Tidak scalable untuk banyak users
❌ Complex dependencies

---

## 📞 Support & Updates

### Logs

Check service logs:
```bash
sudo journalctl -u security-system-web -f
```

Check application logs:
```bash
tail -f ~/riftech-cam-security/logs/*.log
```

### Restart Service

```bash
sudo systemctl restart security-system-web
```

### Update Application

```bash
cd ~/riftech-cam-security
git pull
source venv/bin/activate
pip install -r requirements.txt --upgrade
sudo systemctl restart security-system-web
```

---

## 🎯 Contoh Use Cases

### 1. Home Monitoring

- 1 kamera di living room
- 1 zone untuk pintu depan
- Confidence: 25%
- Model: YOLOv8 Nano
- Alerts via Telegram

### 2. Office Security

- 2 kamera (front & back)
- Multiple zones untuk entry points
- Confidence: 30%
- Model: YOLOv8 Small
- Recording enabled 24/7
- Trusted faces: 10 employees

### 3. Warehouse Monitoring

- 4 kamera
- Large zones untuk aisles
- Confidence: 20%
- Model: YOLOv8 Medium
- Heat map enabled
- Night vision enabled

---

## 📈 Roadmap

### Phase 1 (Current)
- ✅ Basic video streaming
- ✅ All controls from GUI
- ✅ Zone management
- ✅ Detection settings
- ✅ Real-time statistics

### Phase 2 (Future)
- [ ] Multi-camera support
- [ ] Playback recordings dari web
- [ ] Export events ke CSV
- [ ] User authentication
- [ ] Mobile app (React Native)
- [ ] Cloud storage backup
- [ ] AI alerts customization
- [ ] Schedule arming/disarming

### Phase 3 (Advanced)
- [ ] Video analytics dashboard
- [ ] People counting
- [ ] Dwell time tracking
- [ ] Heat map timeline
- [ ] Integration dengan alarm sistem
- [ ] API untuk third-party integrations

---

## 🤝 Contributing

Contributions welcome! Fork dan pull request.

### Development Setup

```bash
cd riftech-cam-security
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install websockets

# Run development server
python web_server.py
```

---


---

## 📞 Support

Untuk issues atau contributions:
- GitHub: https://github.com/riftech22/riftech-cam-security

---

**Selamat menggunakan Riftech Cam Security!** 🎉
