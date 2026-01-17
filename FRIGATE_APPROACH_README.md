# Pendekatan Frigate untuk V380 Split Frame

## Perbandingan: Pendekatan Lama vs Frigate

### Pendekatan Lama (WebSocket + Base64 Images)

**Arsitektur:**
```
Camera RTSP → OpenCV capture → JPEG encode → Base64 → WebSocket → Browser
                     ↓
                 YOLO detection (pada full frame split)
```

**Masalah:**
1. **Inefisien Base64 encoding**: Base64 menambah 33% ukuran data
2. **Tidak ada frame cropping**: YOLO menerima frame split dengan aspect ratio aneh
3. **WebSocket overhead**: Overhead protokol WebSocket
4. **Tidak ada buffering**: Frame drop jika network lambat
5. **YOLO bingung**: Aspect ratio 1280:360 (3.5:1) membuat orang terlihat pendek dan gemuk

### Pendekatan Frigate (FFmpeg Pipeline + HLS)

**Arsitektur:**
```
Camera RTSP → FFmpeg (decode & split) → Queue 1 → YOLO detection
                     ↓                              ↓
              MJPEG pipe stdout             Queue 2 → Web Interface
```

**Keunggulan:**
1. **FFmpeg hardware acceleration**: Support VAAPI, NVIDIA, QSV, dll
2. **Proper frame cropping**: Frame dipisahkan sebelum deteksi
3. **Asynchronous processing**: Capture dan detection di thread terpisah
4. **Queue buffering**: Smooth meskipun load tinggi
5. **Proper aspect ratio**: Setiap split memiliki aspect ratio normal
6. **HLS streaming** (opsional): Standar industri untuk video streaming

## Implementasi V380FFmpegProcessor

### Komponen Utama

1. **FFmpeg Capture Thread**
   - Capture stream RTSP
   - Decode ke MJPEG
   - Split frame menjadi 2 bagian (atas/bawah)
   - Masukkan ke frame queue

2. **YOLO Detection Thread**
   - Ambil frame dari queue
   - Jalankan YOLO detection pada masing-masing split
   - Masukkan hasil ke detection queue

3. **Frame Queues**
   - `frame_queue`: Buffer untuk frame raw (maxsize=10)
   - `detection_queue`: Buffer untuk hasil deteksi (maxsize=10)

### Frame Flow

```
RTSP Stream (1280x720)
         ↓
FFmpeg decode & scale (640x360)
         ↓
Split horizontally:
  - Top frame: (0,0) to (640,360)
  - Bottom frame: (0,360) to (640,720)
         ↓
Queue 1 → YOLO detection → Queue 2 → Web Interface
```

## Konfigurasi FFmpeg

### Command Dasar

```bash
ffmpeg -rtsp_transport tcp \
       -stimeout 5000000 \
       -i rtsp://user:pass@ip:554/live \
       -vf "fps=5,scale=640:360" \
       -f image2pipe \
       -vcodec mjpeg \
       -q:v 2 \
       -
```

### Penjelasan Parameters

| Parameter | Nilai | Penjelasan |
|-----------|-------|-----------|
| `-rtsp_transport tcp` | tcp | Gunakan TCP untuk RTSP (lebih stabil) |
| `-stimeout` | 5000000 | 5 second timeout |
| `-vf fps=5` | 5 | Limit ke 5 FPS untuk deteksi |
| `-vf scale=640:360` | 640:360 | Scale untuk efisiensi |
| `-f image2pipe` | - | Output sebagai pipe MJPEG |
| `-vcodec mjpeg` | mjpeg | Codec MJPEG |
| `-q:v 2` | 2 | Kualitas JPEG (2 = baik) |

## Penggunaan

### Standalone Test

```python
from v380_ffmpeg_pipeline import V380FFmpegProcessor

# Konfigurasi
processor = V380FFmpegProcessor(
    rtsp_url="rtsp://admin:admin@192.168.1.108:554/live",
    model_path="yolov8s.pt",
    detect_fps=5,
    device="cpu",
    conf_threshold=0.25,
    iou_threshold=0.45
)

# Start
processor.start()

# Get detections
while True:
    result = processor.get_latest_detection()
    if result:
        top_detections = result['top_detections']
        bottom_detections = result['bottom_detections']
        # Process detections...

# Stop
processor.stop()
```

### Integrasi dengan Web Server

```python
from flask import Flask, Response, jsonify
from v380_ffmpeg_pipeline import V380FFmpegProcessor
import cv2
import base64
import json

app = Flask(__name__)

# Initialize processor
processor = V380FFmpegProcessor(
    rtsp_url="rtsp://admin:admin@192.168.1.108:554/live",
    model_path="yolov8s.pt",
    detect_fps=5
)

@app.route('/api/stream')
def stream():
    """Streaming frames dengan detections (JPEG format)."""
    def generate():
        while True:
            result = processor.get_latest_detection()
            if result:
                # Draw detections
                top_frame = processor.draw_detections(
                    result['top_frame'],
                    result['top_detections'],
                    "Top Camera"
                )
                bottom_frame = processor.draw_detections(
                    result['bottom_frame'],
                    result['bottom_detections'],
                    "Bottom Camera"
                )
                
                # Stack frames
                display_frame = np.vstack([top_frame, bottom_frame])
                
                # Encode to JPEG
                _, buffer = cv2.imencode('.jpg', display_frame)
                frame_bytes = buffer.tobytes()
                
                # Yield frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/detections')
def get_detections():
    """Get JSON detection results."""
    result = processor.get_latest_detection()
    if result:
        return jsonify({
            'timestamp': result['timestamp'],
            'top_count': len(result['top_detections'].boxes) if result['top_detections'] else 0,
            'bottom_count': len(result['bottom_detections'].boxes) if result['bottom_detections'] else 0,
            'top_detections': format_detections(result['top_detections']),
            'bottom_detections': format_detections(result['bottom_detections'])
        })
    return jsonify({'status': 'no_detection'})

if __name__ == '__main__':
    processor.start()
    app.run(host='0.0.0.0', port=5000)
```

## Keunggulan vs Pendekatan Lama

| Aspek | Pendekatan Lama | Pendekatan Frigate |
|-------|----------------|-------------------|
| **Frame Capture** | OpenCV VideoCapture | FFmpeg (hardware accel) |
| **Frame Split** | Python array slicing | FFmpeg filters |
| **Frame Queue** | Tidak ada | Queue-based buffering |
| **Detection FPS** | 1-2 FPS (CPU bound) | 5+ FPS (async) |
| **Aspect Ratio** | 3.5:1 (distorsi) | 16:9 (normal) |
| **Memory Usage** | Tinggi (base64) | Rendah (binary) |
| **Latency** | Tinggi | Rendah |
| **Scalability** | Buruk | Bagus (queue sizing) |

## Rekomendasi untuk Implementasi Produksi

1. **Gunakan Hardware Acceleration**
   ```python
   processor = V380FFmpegProcessor(
       rtsp_url=...,
       device="cuda"  # atau "cpu" jika tidak ada GPU
   )
   ```

2. **Tuning FPS berdasarkan CPU/GPU**
   - CPU only: 2-3 FPS
   - GPU NVIDIA: 5-10 FPS
   - Jetson/Raspberry Pi: 3-5 FPS

3. **Monitor Queue Sizes**
   ```python
   print(f"Frame queue: {processor.frame_queue.qsize()}")
   print(f"Detection queue: {processor.detection_queue.qsize()}")
   ```

4. **Implement Error Recovery**
   - Restart FFmpeg jika crash
   - Restart detection thread jika hang
   - Monitor FPS dan queue sizes

5. **Optimize untuk Web Interface**
   - Gunakan MJPEG streaming untuk live view
   - Gunakan JSON API untuk detection data
   - Implement HLS untuk playback (opsional)

## Migrasi dari Pendekatan Lama

1. **Ganti OpenCV VideoCapture dengan FFmpeg**
   ```python
   # Lama
   cap = cv2.VideoCapture(rtsp_url)
   
   # Baru
   processor = V380FFmpegProcessor(rtsp_url)
   ```

2. **Hapus Base64 Encoding**
   ```python
   # Lama
   frame_base64 = base64.b64encode(jpeg_bytes).decode()
   
   # Baru
   # Tidak perlu base64, kirim binary langsung
   ```

3. **Implement Queue-based Architecture**
   ```python
   # Lama: Synchronous
   while True:
       ret, frame = cap.read()
       detections = model(frame)
   
   # Baru: Asynchronous dengan queue
   processor.start()
   while True:
       result = processor.get_latest_detection()
   ```

## Kesimpulan

Pendekatan Frigate menggunakan FFmpeg pipeline memberikan:
- ✅ Performance lebih baik
- ✅ Hardware acceleration support
- ✅ Asynchronous processing
- ✅ Queue buffering untuk smooth operation
- ✅ Proper aspect ratio untuk YOLO detection
- ✅ Scalable architecture

Ini adalah pendekatan industri yang digunakan oleh Frigate dan sistem video surveillance profesional.
