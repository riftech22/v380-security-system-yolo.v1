#!/usr/bin/env python3
"""Web Server untuk Security System - Robust Version for Ubuntu Server."""

import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
import time
import threading
import logging
import sys
from typing import Set, Dict, Optional
import os
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Import security system modules with error handling
try:
    from config import Config, AlertType
    from detectors import PersonDetector, FaceRecognitionEngine, MotionDetector, DetectionThread
    from database import DatabaseManager
    from utils import MultiZoneManager
    MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"[Import] Some modules not available: {e}")
    logging.warning("[Import] Running in demo mode without detection features")
    MODULES_AVAILABLE = False


class SecurityWebSystem:
    """Backend Security System untuk web interface."""
    
    def __init__(self):
        self.config = None
        self.db = None
        self.zone_manager = None
        
        # Detectors
        self.person_detector = None
        self.face_engine = None
        self.motion_detector = None
        self.detection_thread = None
        
        # Camera
        self.cap = None
        self.running = False
        self.camera_available = False
        
        # State
        self.is_armed = False
        self.is_recording = False
        self.is_muted = False
        
        # Features
        self.enable_skeleton = False
        self.enable_face = True
        self.enable_motion = True
        self.enable_heatmap = False
        self.night_vision = False
        
        # Detection settings
        self.confidence = 0.25
        self.model_name = 'yolov8n.pt'
        
        # Statistics
        self.person_count = 0
        self.alert_count = 0
        self.breach_active = False
        self.breach_start_time = 0
        self.trusted_detected = False
        self.trusted_name = ""
        
        # Frame
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.last_frame_time = 0
        self.consecutive_failures = 0
        self.max_consecutive_failures = 30
        
        # Video recording
        self.video_writer = None
        
        # Demo mode flag
        self.demo_mode = not MODULES_AVAILABLE
        
        if not self.demo_mode:
            try:
                self.config = Config()
                self.db = DatabaseManager(self.config)
                self.zone_manager = MultiZoneManager()
            except Exception as e:
                logging.error(f"[Init] Error initializing modules: {e}")
                self.demo_mode = True
    
    def start_camera(self):
        """Start camera."""
        logging.info("[Camera] Initializing...")
        
        try:
            # Coba koneksi ke camera source (RTSP atau USB)
            camera_source = Config.CAMERA_SOURCE if self.config else 0
            
            self.cap = cv2.VideoCapture(camera_source)
            
            if self.cap.isOpened():
                # Coba baca frame
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    logging.info(f"[Camera] Connected to camera: {camera_source}")
                    # Set backend dan buffer size untuk RTSP
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    self.camera_available = True
                    return True
                else:
                    self.cap.release()
            else:
                if self.cap:
                    self.cap.release()
        except Exception as e:
            logging.warning(f"[Camera] Error: {e}")
        
        logging.warning("[Camera] Warning: No camera detected, using demo mode")
        self.camera_available = False
        return False
    
    def start_detection(self):
        """Start detection modules."""
        if self.demo_mode:
            logging.info("[Detection] Demo mode - skipping detection modules")
            return True
        
        logging.info("[Detection] Loading modules...")
        
        try:
            self.person_detector = PersonDetector(self.config)
            self.person_detector.set_confidence(self.confidence)
            
            self.face_engine = FaceRecognitionEngine(self.config)
            self.motion_detector = MotionDetector(self.config)
            
            self.detection_thread = DetectionThread(self.person_detector, self.motion_detector)
            self.detection_thread.draw_skeleton = self.enable_skeleton
            self.detection_thread.start()
            
            logging.info("[Detection] Modules loaded")
            return True
        except Exception as e:
            logging.error(f"[Detection] Error: {e}")
            logging.warning("[Detection] Running without detection features")
            return True  # Continue without detection
    
    def capture_frame(self):
        """Capture frame dengan RTSP fix untuk mengatasi stuck frame."""
        if self.camera_available and self.cap and self.cap.isOpened():
            try:
                # Clear buffer untuk RTSP (hindari frame lama)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                # Baca frame dengan retry
                max_retries = 3
                frame = None
                
                for retry in range(max_retries):
                    ret, temp_frame = self.cap.read()
                    
                    if ret and temp_frame is not None and isinstance(temp_frame, np.ndarray):
                        frame = temp_frame
                        
                        # Cek jika frame valid (bukan frame kosong)
                        if frame.shape[0] > 0 and frame.shape[1] > 0:
                            # Submit ke detection thread
                            if self.detection_thread and not self.demo_mode:
                                self.detection_thread.submit(frame)
                            
                            self.consecutive_failures = 0
                            return frame
                        else:
                            logging.warning(f"[Capture] Invalid frame shape: {frame.shape}")
                    else:
                        if retry < max_retries - 1:
                            logging.warning(f"[Capture] Retry {retry + 1}/{max_retries}")
                            time.sleep(0.05)
                
                # Jika semua retry gagal
                if frame is None:
                    self.consecutive_failures += 1
                    if self.consecutive_failures > self.max_consecutive_failures:
                        logging.error(f"[Capture] Too many failures, reconnecting...")
                        # Coba reconnect kamera
                        self.cap.release()
                        time.sleep(1)
                        self.cap = cv2.VideoCapture(Config.CAMERA_SOURCE if self.config else 0)
                        if self.cap.isOpened():
                            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                            self.cap.set(cv2.CAP_PROP_FPS, 30)
                            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            self.consecutive_failures = 0
                            return self.capture_frame()
            except Exception as e:
                logging.error(f"[Capture] Error: {e}")
                self.consecutive_failures += 1
        
        # Use last frame or create demo frame
        with self.frame_lock:
            if self.current_frame is not None:
                # Return frame baru dengan timestamp update untuk mencegah cache di browser
                return self.current_frame.copy()
        
        # Create demo frame
        return self.create_demo_frame()
    
    def create_demo_frame(self):
        """Create demo frame for testing."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Add background
        frame[:, :] = (30, 30, 40)
        
        # Add text
        cv2.putText(frame, "RIFTECH CAM SECURITY", (340, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(frame, "Demo Mode - No Camera Detected", (320, 260),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)
        
        # Add animated elements
        t = time.time()
        x = int(640 + 200 * np.sin(t))
        y = int(360 + 100 * np.cos(t))
        cv2.circle(frame, (x, y), 20, (0, 255, 0), -1)
        
        cv2.putText(frame, f"Server Time: {time.strftime('%H:%M:%S')}", (400, 450),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame dengan detection dan overlays."""
        if frame is None:
            return self.create_demo_frame()
        
        h, w = frame.shape[:2]
        output = frame.copy()
        
        # Timestamp
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(output, ts, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Get detection results
        if not self.demo_mode and self.detection_thread:
            results = self.detection_thread.get_results()
            persons = results.get('persons', [])
            has_motion = results.get('motion', False)
            motion_regions = results.get('motion_regions', [])
        else:
            persons = []
            has_motion = False
            motion_regions = []
        
        self.person_count = len(persons)
        
        # Night vision
        if self.night_vision:
            output = cv2.convertScaleAbs(output, alpha=1.5, beta=30)
            gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            output[:, :, 1] = np.clip(output[:, :, 1] * 1.3, 0, 255).astype(np.uint8)
        
        # Heat map
        if self.enable_heatmap and self.motion_detector and not self.demo_mode:
            try:
                hm = self.motion_detector.get_heat_map()
                if hm is not None:
                    if hm.shape[:2] != (h, w):
                        hm = cv2.resize(hm, (w, h))
                    hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
                    output = cv2.addWeighted(output, 0.7, hm_color, 0.3, 0)
            except Exception as e:
                logging.error(f"[Heatmap] Error: {e}")
        
        # Motion boxes
        if self.enable_motion:
            for mx1, my1, mx2, my2 in motion_regions:
                cv2.rectangle(output, (mx1, my1), (mx2, my2), (0, 165, 255), 1)
        
        # Draw persons
        for person in persons:
            x1, y1, x2, y2 = person.bbox
            color = (0, 0, 255) if self.breach_active else (0, 255, 0)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output, f"Person {person.confidence:.0%}", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw zones
        if self.zone_manager:
            breached_ids = []
            if self.breach_active:
                for zone in self.zone_manager.zones:
                    breached_ids.append(zone.zone_id)
            output = self.zone_manager.draw_all(output, breached_ids, self.is_armed)
        
        # Recording indicator
        if self.is_recording:
            cv2.circle(output, (w - 25, 25), 10, (0, 0, 255), -1)
            cv2.putText(output, "REC", (w - 65, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Armed indicator
        if self.is_armed:
            cv2.putText(output, "ARMED", (w - 85, h - 15), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return output
    
    def toggle_arm(self, armed: bool):
        """Toggle arm/disarm."""
        self.is_armed = armed
        logging.info(f"[System] Armed: {armed}")
    
    def toggle_record(self, recording: bool):
        """Toggle recording."""
        self.is_recording = recording
        logging.info(f"[Record] Recording: {recording}")
    
    def toggle_mute(self, muted: bool):
        """Toggle mute."""
        self.is_muted = muted
        logging.info(f"[Audio] Muted: {muted}")
    
    def take_snapshot(self) -> str:
        """Take snapshot."""
        with self.frame_lock:
            if self.current_frame is None:
                frame = self.create_demo_frame()
            else:
                frame = self.current_frame.copy()
        
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = f"snapshots/snap_{ts}.jpg"
        try:
            cv2.imwrite(path, frame)
            logging.info(f"[Snapshot] Saved: {path}")
        except Exception as e:
            logging.error(f"[Snapshot] Error: {e}")
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return base64.b64encode(buffer).decode('utf-8')
    
    def set_confidence(self, conf: float):
        """Set confidence level."""
        self.confidence = conf
        if self.person_detector:
            self.person_detector.set_confidence(conf)
        logging.info(f"[Detection] Confidence: {conf}")
    
    def set_model(self, model_name: str):
        """Set YOLO model."""
        self.model_name = model_name
        if self.person_detector:
            self.person_detector.change_model(model_name)
        logging.info(f"[Detection] Model: {model_name}")
    
    def toggle_skeleton(self, enabled: bool):
        """Toggle skeleton detection."""
        self.enable_skeleton = enabled
        if self.detection_thread:
            self.detection_thread.draw_skeleton = enabled
        logging.info(f"[Detection] Skeleton: {enabled}")
    
    def reload_faces(self):
        """Reload trusted faces."""
        if self.face_engine:
            self.face_engine.reload_faces()
            logging.info(f"[Faces] Reloaded")
        else:
            logging.info("[Faces] Face engine not available in demo mode")
    
    def create_zone(self):
        """Create new zone."""
        if self.zone_manager:
            self.zone_manager.create_zone()
            logging.info(f"[Zone] Created: Total {self.zone_manager.get_zone_count()}")
        else:
            logging.info("[Zone] Zone manager not available in demo mode")
    
    def add_zone_point(self, x: int, y: int):
        """Add point to active zone."""
        if self.zone_manager:
            zone = self.zone_manager.get_active_zone()
            if zone:
                zone.add_point(x, y)
                logging.info(f"[Zone] Point added: ({x}, {y})")
    
    def clear_zones(self):
        """Clear all zones."""
        if self.zone_manager:
            self.zone_manager.delete_all_zones()
        self.breach_active = False
        logging.info("[Zone] Cleared all zones")
    
    def stop(self):
        """Stop system."""
        self.running = False
        if self.cap:
            self.cap.release()
        if self.detection_thread:
            self.detection_thread.stop()
        if self.video_writer:
            self.video_writer.release()
        logging.info("[System] Stopped")


class SecurityWebServer:
    """WebSocket server untuk web interface."""
    
    def __init__(self):
        self.system = SecurityWebSystem()
        self.clients: Set = set()
        self.running = False
        
    async def handle_client(self, websocket):
        """Handle WebSocket connection."""
        client_addr = websocket.remote_address
        logging.info(f"[WebSocket] Client connected: {client_addr}")
        self.clients.add(websocket)
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_command(data, websocket)
                except Exception as e:
                    logging.error(f"[WebSocket] Error handling message: {e}")
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logging.error(f"[WebSocket] Error: {e}")
        finally:
            self.clients.discard(websocket)
            logging.info(f"[WebSocket] Client disconnected: {client_addr}")
    
    async def handle_command(self, data: Dict, websocket):
        """Handle command dari client."""
        cmd_type = data.get('type')
        
        if cmd_type == 'get_status':
            status = {
                'type': 'status',
                'armed': self.system.is_armed,
                'recording': self.system.is_recording,
                'muted': self.system.is_muted,
                'persons': self.system.person_count,
                'alerts': self.system.alert_count,
                'breach_active': self.system.breach_active,
                'breach_duration': int(time.time() - self.system.breach_start_time) if self.system.breach_active else 0,
                'confidence': self.system.confidence,
                'model': self.system.model_name,
                'skeleton': self.system.enable_skeleton,
                'face': self.system.enable_face,
                'motion': self.system.enable_motion,
                'heatmap': self.system.enable_heatmap,
                'night_vision': self.system.night_vision,
                'zones': self.system.zone_manager.get_zone_count() if self.system.zone_manager else 0,
                'faces': len(self.system.face_engine.known_names) if self.system.face_engine else 0,
                'clients': len(self.clients),
                'demo_mode': self.system.demo_mode,
                'camera_available': self.system.camera_available
            }
            await websocket.send(json.dumps(status))
        
        elif cmd_type == 'toggle_arm':
            self.system.toggle_arm(data.get('value', False))
        
        elif cmd_type == 'toggle_record':
            self.system.toggle_record(data.get('value', False))
        
        elif cmd_type == 'toggle_mute':
            self.system.toggle_mute(data.get('value', False))
        
        elif cmd_type == 'snapshot':
            snapshot = self.system.take_snapshot()
            if snapshot:
                await websocket.send(json.dumps({
                    'type': 'snapshot',
                    'data': snapshot
                }))
        
        elif cmd_type == 'set_confidence':
            self.system.set_confidence(data.get('value', 0.25))
        
        elif cmd_type == 'set_model':
            self.system.set_model(data.get('value', 'yolov8n.pt'))
        
        elif cmd_type == 'toggle_skeleton':
            self.system.toggle_skeleton(data.get('value', False))
        
        elif cmd_type == 'toggle_face':
            self.system.enable_face = data.get('value', True)
            logging.info(f"[Detection] Face: {self.system.enable_face}")
        
        elif cmd_type == 'toggle_motion':
            self.system.enable_motion = data.get('value', True)
            logging.info(f"[Detection] Motion: {self.system.enable_motion}")
        
        elif cmd_type == 'toggle_heatmap':
            self.system.enable_heatmap = data.get('value', False)
            logging.info(f"[Detection] Heatmap: {self.system.enable_heatmap}")
        
        elif cmd_type == 'toggle_night_vision':
            self.system.night_vision = data.get('value', False)
            logging.info(f"[Display] Night Vision: {self.system.night_vision}")
        
        elif cmd_type == 'create_zone':
            self.system.create_zone()
        
        elif cmd_type == 'add_zone_point':
            x, y = data.get('x', 0), data.get('y', 0)
            self.system.add_zone_point(x, y)
        
        elif cmd_type == 'clear_zones':
            self.system.clear_zones()
        
        elif cmd_type == 'reload_faces':
            self.system.reload_faces()
        
        elif cmd_type == 'get_zones':
            zones_data = []
            if self.system.zone_manager:
                for zone in self.system.zone_manager.zones:
                    zones_data.append({
                        'id': zone.zone_id,
                        'name': zone.name,
                        'points': zone.points,
                        'complete': zone.is_complete
                    })
            await websocket.send(json.dumps({
                'type': 'zones',
                'zones': zones_data
            }))
    
    async def broadcast_frame(self, processed_frame: np.ndarray):
        """Broadcast frame ke semua clients."""
        try:
            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            message = json.dumps({
                'type': 'frame',
                'timestamp': time.time(),
                'data': frame_b64
            })
            
            if self.clients:
                await asyncio.gather(
                    *[client.send(message) for client in self.clients],
                    return_exceptions=True
                )
        except Exception as e:
            logging.error(f"[WebSocket] Broadcast error: {e}")
    
    async def start(self):
        """Start server."""
        logging.info("[Server] Starting Security System...")
        
        try:
            # Start camera (non-fatal if fails)
            self.system.start_camera()
            
            # Start detection (non-fatal if fails)
            self.system.start_detection()
            
            self.system.running = True
            
            # Start frame capture loop
            async def capture_loop():
                frame_count = 0
                while self.system.running:
                    try:
                        frame = self.system.capture_frame()
                        if frame is not None:
                            processed = self.system.process_frame(frame)
                            await self.broadcast_frame(processed)
                            frame_count += 1
                            if frame_count % 30 == 0:  # Log setiap 30 frame (~1 detik)
                                logging.info(f"[Capture] Processed {frame_count} frames, {len(self.clients)} clients")
                    except Exception as e:
                        logging.error(f"[Capture] Error: {e}")
                    await asyncio.sleep(1/30)  # 30 FPS
            
            asyncio.create_task(capture_loop())
            
            # Start WebSocket server
            async with websockets.serve(
                self.handle_client,
                "0.0.0.0",
                8765,
                ping_interval=30,
                ping_timeout=60,
                max_size=10 * 1024 * 1024
            ):
                logging.info("[Server] WebSocket server started on ws://0.0.0.0:8765")
                if self.system.demo_mode:
                    logging.info("[Server] Running in DEMO MODE (no camera/detection)")
                else:
                    logging.info("[Server] Running in NORMAL MODE")
                logging.info("[Server] Access web interface at: http://YOUR_SERVER_IP:8080/web.html")
                
                # Keep running
                while self.system.running:
                    await asyncio.sleep(1)
            
            return True
        except Exception as e:
            logging.error(f"[Server] Fatal error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def stop(self):
        """Stop server."""
        self.system.stop()


async def main():
    server = SecurityWebServer()
    
    try:
        success = await server.start()
        if not success:
            logging.error("[Server] Failed to start server")
            sys.exit(1)
    except KeyboardInterrupt:
        logging.info("\n[Server] Shutting down...")
        server.stop()
    except Exception as e:
        logging.error(f"[Server] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        server.stop()
        sys.exit(1)


if __name__ == "__main__":
    logging.info("Starting Security Web Server (Robust Mode)...")
    asyncio.run(main())
