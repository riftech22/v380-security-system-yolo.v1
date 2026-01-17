#!/usr/bin/env python3
"""
V380 Split Frame Processing - Frigate-style Implementation
===========================================================
Menggunakan FFmpeg pipeline seperti Frigate untuk menangani kamera V380 split.

Pendekatan:
1. FFmpeg capture stream dari RTSP
2. FFmpeg crop frame menjadi 2 bagian (atas/bawah)
3. YOLO detection pada masing-masing frame yang sudah dipisahkan
4. HLS streaming untuk web interface (opsional, atau gunakan JPEG snapshots)
"""

import cv2
import numpy as np
import subprocess as sp
import threading
import queue
import time
from datetime import datetime
from typing import Optional, Tuple, List

# YOLO imports
from ultralytics import YOLO

class V380FFmpegProcessor:
    """Processor untuk kamera V380 split frame menggunakan FFmpeg pipeline."""
    
    def __init__(
        self,
        rtsp_url: str,
        model_path: str = "yolov8s.pt",
        detect_fps: int = 5,
        device: str = "cpu",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        """
        Inisialisasi processor.
        
        Args:
            rtsp_url: URL RTSP kamera
            model_path: Path ke model YOLO
            detect_fps: FPS untuk deteksi
            device: Device untuk inference (cpu/cuda)
            conf_threshold: Threshold confidence
            iou_threshold: Threshold IoU untuk NMS
        """
        self.rtsp_url = rtsp_url
        self.detect_fps = detect_fps
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Initialize YOLO model
        print(f"[INFO] Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        self.model.to(device)
        print(f"[INFO] Model loaded on {device}")
        
        # Frame dimensions (default 1280x720 untuk V380)
        self.frame_width = 1280
        self.frame_height = 720
        self.split_height = self.frame_height // 2  # 360 per split
        
        # Frame queues
        self.frame_queue = queue.Queue(maxsize=10)
        self.detection_queue = queue.Queue(maxsize=10)
        
        # Control flags
        self.running = False
        self.ffmpeg_process = None
        self.capture_thread = None
        self.detection_thread = None
        
        # Statistics
        self.capture_fps = 0
        self.detection_fps = 0
        self.frame_count = 0
        self.detection_count = 0
        
    def build_ffmpeg_command(self, output_width: int = 640, output_height: int = 360) -> List[str]:
        """
        Build FFmpeg command untuk capture RTSP dan crop frame menjadi 2 bagian.
        
        Args:
            output_width: Lebar output per split
            output_height: Tinggi output per split
            
        Returns:
            List of FFmpeg command arguments
        """
        cmd = [
            'ffmpeg',
            # Input settings
            '-rtsp_transport', 'tcp',
            '-stimeout', '5000000',  # 5 second timeout
            '-i', self.rtsp_url,
            
            # Video filters
            '-vf', f'fps={self.detect_fps},scale={output_width}:{output_height}',
            
            # Output settings - MJPEG stream ke stdout
            '-f', 'image2pipe',
            '-vcodec', 'mjpeg',
            '-q:v', '2',  # Quality 2 (good quality)
            '-'
        ]
        return cmd
    
    def capture_frames(self):
        """Capture frames dari FFmpeg dan masukkan ke queue."""
        print("[CAPTURE] Starting capture thread...")
        
        try:
            # Start FFmpeg
            cmd = self.build_ffmpeg_command(
                output_width=640,  # Width per split (setengah dari 1280 untuk efisiensi)
                output_height=360  # Height per split (setengah dari 720)
            )
            
            print(f"[CAPTURE] FFmpeg command: {' '.join(cmd)}")
            
            self.ffmpeg_process = sp.Popen(
                cmd,
                stdout=sp.PIPE,
                stderr=sp.DEVNULL,
                stdin=sp.DEVNULL,
                bufsize=10**8
            )
            
            fps_counter = 0
            fps_time = time.time()
            
            # Read frame bytes from FFmpeg
            frame_bytes = b''
            frame_size = 0
            
            while self.running:
                try:
                    # Read data from FFmpeg
                    data = self.ffmpeg_process.stdout.read(1024)
                    
                    if not data:
                        print("[CAPTURE] No more data from FFmpeg")
                        break
                        
                    frame_bytes += data
                    
                    # Check if we have a complete frame (JPEG end marker)
                    # JPEG ends with FF D9
                    if b'\xff\xd9' in frame_bytes:
                        # Extract frame
                        end_marker = frame_bytes.find(b'\xff\xd9') + 2
                        jpeg_data = frame_bytes[:end_marker]
                        frame_bytes = frame_bytes[end_marker:]
                        
                        # Decode JPEG to numpy array
                        frame = cv2.imdecode(
                            np.frombuffer(jpeg_data, dtype=np.uint8),
                            cv2.IMREAD_COLOR
                        )
                        
                        if frame is not None:
                            # Split frame into top and bottom
                            frame_height = frame.shape[0]
                            split_point = frame_height // 2
                            
                            top_frame = frame[:split_point, :, :]
                            bottom_frame = frame[split_point:, :, :]
                            
                            # Put into queue
                            try:
                                self.frame_queue.put({
                                    'timestamp': time.time(),
                                    'top': top_frame,
                                    'bottom': bottom_frame
                                }, block=False)
                                
                                fps_counter += 1
                                
                            except queue.Full:
                                # Skip frame if queue is full
                                pass
                    
                    # Calculate FPS
                    if time.time() - fps_time >= 1.0:
                        self.capture_fps = fps_counter
                        fps_counter = 0
                        fps_time = time.time()
                        print(f"[CAPTURE] FPS: {self.capture_fps}")
                
                except Exception as e:
                    print(f"[CAPTURE] Error: {e}")
                    if self.running:
                        time.sleep(0.1)
                    else:
                        break
                        
        except Exception as e:
            print(f"[CAPTURE] Fatal error: {e}")
        
        finally:
            if self.ffmpeg_process:
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait()
            print("[CAPTURE] Capture thread stopped")
    
    def process_detections(self):
        """Process detections pada frame yang sudah dipisahkan."""
        print("[DETECT] Starting detection thread...")
        
        detection_counter = 0
        detection_time = time.time()
        
        while self.running:
            try:
                # Get frame from queue (with timeout)
                frame_data = self.frame_queue.get(timeout=1.0)
                
                timestamp = frame_data['timestamp']
                top_frame = frame_data['top']
                bottom_frame = frame_data['bottom']
                
                # Process top frame (kamera fixed)
                top_results = self.model(
                    top_frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
                
                # Process bottom frame (kamera PTZ)
                bottom_results = self.model(
                    bottom_frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
                
                # Put results into queue
                result_data = {
                    'timestamp': timestamp,
                    'top_detections': top_results[0],
                    'bottom_detections': bottom_results[0],
                    'top_frame': top_frame,
                    'bottom_frame': bottom_frame
                }
                
                try:
                    self.detection_queue.put(result_data, block=False)
                    detection_counter += 1
                except queue.Full:
                    pass
                
                # Calculate detection FPS
                if time.time() - detection_time >= 1.0:
                    self.detection_fps = detection_counter
                    detection_counter = 0
                    detection_time = time.time()
                    print(f"[DETECT] FPS: {self.detection_fps}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[DETECT] Error: {e}")
                if self.running:
                    time.sleep(0.1)
                else:
                    break
        
        print("[DETECT] Detection thread stopped")
    
    def start(self):
        """Start processing."""
        print("[MAIN] Starting V380 FFmpeg processor...")
        
        self.running = True
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.capture_thread.start()
        
        # Wait a bit for frames to accumulate
        time.sleep(2)
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self.process_detections, daemon=True)
        self.detection_thread.start()
        
        print("[MAIN] Processor started")
    
    def stop(self):
        """Stop processing."""
        print("[MAIN] Stopping processor...")
        
        self.running = False
        
        # Wait for threads
        if self.capture_thread:
            self.capture_thread.join(timeout=5)
        
        if self.detection_thread:
            self.detection_thread.join(timeout=5)
        
        # Terminate FFmpeg
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
            self.ffmpeg_process.wait(timeout=5)
        
        # Clear queues
        while not self.frame_queue.empty():
            self.frame_queue.get()
        
        while not self.detection_queue.empty():
            self.detection_queue.get()
        
        print("[MAIN] Processor stopped")
    
    def get_latest_detection(self):
        """Get latest detection result."""
        try:
            return self.detection_queue.get_nowait()
        except queue.Empty:
            return None
    
    def draw_detections(self, frame, results, camera_name: str = "Camera"):
        """
        Draw detection boxes pada frame.
        
        Args:
            frame: OpenCV image
            results: YOLO results
            camera_name: Name of camera for label
            
        Returns:
            Frame dengan detection boxes
        """
        frame_copy = frame.copy()
        
        # Draw camera name
        cv2.putText(
            frame_copy,
            camera_name,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        
        if results is None or len(results.boxes) == 0:
            return frame_copy
        
        for box in results.boxes:
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            
            # Draw box
            cv2.rectangle(
                frame_copy,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )
            
            # Draw label
            label = f"{self.model.names[cls]} {conf:.2f}"
            cv2.putText(
                frame_copy,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        return frame_copy


def test_v380_ffmpeg_pipeline():
    """Test FFmpeg pipeline dengan kamera V380."""
    
    # Configuration
    rtsp_url = "rtsp://admin:admin@192.168.1.108:554/live"  # Ganti dengan URL kamera Anda
    
    # Initialize processor
    processor = V380FFmpegProcessor(
        rtsp_url=rtsp_url,
        model_path="yolov8s.pt",
        detect_fps=5,
        device="cpu",  # Ganti ke "cuda" jika ada GPU NVIDIA
        conf_threshold=0.25,
        iou_threshold=0.45
    )
    
    # Start processing
    processor.start()
    
    print("\n[TEST] Starting test... Press 'q' to quit")
    
    try:
        while True:
            # Get latest detection
            result = processor.get_latest_detection()
            
            if result:
                timestamp = result['timestamp']
                top_frame = result['top_frame']
                bottom_frame = result['bottom_frame']
                top_detections = result['top_detections']
                bottom_detections = result['bottom_detections']
                
                # Draw detections
                top_frame_drawn = processor.draw_detections(
                    top_frame,
                    top_detections,
                    "Top Camera (Fixed)"
                )
                
                bottom_frame_drawn = processor.draw_detections(
                    bottom_frame,
                    bottom_detections,
                    "Bottom Camera (PTZ)"
                )
                
                # Stack frames vertically for display
                display_frame = np.vstack([top_frame_drawn, bottom_frame_drawn])
                
                # Display FPS info
                cv2.putText(
                    display_frame,
                    f"Capture FPS: {processor.capture_fps} | Detect FPS: {processor.detection_fps}",
                    (10, display_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
                
                # Resize untuk display jika perlu
                if display_frame.shape[0] > 720:
                    scale = 720 / display_frame.shape[0]
                    display_frame = cv2.resize(
                        display_frame,
                        (int(display_frame.shape[1] * scale), 720)
                    )
                
                # Show frame
                cv2.imshow('V380 Split Detection', display_frame)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            else:
                # Wait a bit if no detection available
                time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n[TEST] Interrupted by user")
    
    finally:
        # Stop processor
        processor.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_v380_ffmpeg_pipeline()
