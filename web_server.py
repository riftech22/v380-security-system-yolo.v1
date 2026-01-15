#!/usr/bin/env python3
"""Web Server untuk Security System dengan Full Controls."""

import asyncio
import websockets
import json
import base64
import cv2
import numpy as np
import time
import threading
from typing import Set, Dict, Optional
import os
from pathlib import Path

# Import security system modules
from config import Config, AlertType
from detectors import PersonDetector, FaceRecognitionEngine, MotionDetector, DetectionThread
from database import DatabaseManager
from utils import MultiZoneManager


class SecurityWebSystem:
    """Backend Security System untuk web interface."""
    
    def __init__(self):
        self.config = Config()
        self.db = DatabaseManager(self.config)
        self.zone_manager = MultiZoneManager()
        
        # Detectors
        self.person_detector = None
        self.face_engine = None
        self.motion_detector = None
        self.detection_thread = None
        
        # Camera
        self.cap = None
        self.running = False
        
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
        
        # Video recording
        self.video_writer = None
        
    def start_camera(self):
        """Start camera."""
        print("[Camera] Initializing...")
        
        for cam_id in range(3):
            self.cap = cv2.VideoCapture(cam_id)
            if self.cap.isOpened():
                ret, _ = self.cap.read()
                if ret:
                    print(f"[Camera] Connected to camera {cam_id}")
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    return True
                else:
                    self.cap.release()
        
        print("[Camera] Warning: No camera detected")
        return False
    
    def start_detection(self):
        """Start detection modules."""
        print("[Detection] Loading modules...")
        
        try:
            self.person_detector = PersonDetector(self.config)
            self.person_detector.set_confidence(self.confidence)
            
            self.face_engine = FaceRecognitionEngine(self.config)
            self.motion_detector = MotionDetector(self.config)
            
            self.detection_thread = DetectionThread(self.person_detector, self.motion_detector)
            self.detection_thread.draw_skeleton = self.enable_skeleton
            self.detection_thread.start()
            
            print("[Detection] Modules loaded")
            return True
        except Exception as e:
            print(f"[Detection] Error: {e}")
            return False
    
    def capture_frame(self):
        """Capture frame dan process."""
        if not self.cap or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if ret and frame is not None:
            with self.frame_lock:
                self.current_frame = frame.copy()
            
            # Submit ke detection thread
            if self.detection_thread:
                self.detection_thread.submit(frame)
            
            return frame
        return None
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame dengan detection dan overlays."""
        if frame is None:
            return np.zeros((720, 1280, 3), dtype=np.uint8)
        
        h, w = frame.shape[:2]
        output = frame.copy()
        
        # Timestamp
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(output, ts, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Get detection results
        results = self.detection_thread.get_results() if self.detection_thread else {}
        persons = results.get('persons', [])
        has_motion = results.get('motion', False)
        motion_regions = results.get('motion_regions', [])
        
        self.person_count = len(persons)
        
        # Night vision
        if self.night_vision:
            output = cv2.convertScaleAbs(output, alpha=1.5, beta=30)
            gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            output[:, :, 1] = np.clip(output[:, :, 1] * 1.3, 0, 255).astype(np.uint8)
        
        # Heat map
        if self.enable_heatmap and self.motion_detector:
            hm = self.motion_detector.get_heat_map()
            if hm is not None:
                if hm.shape[:2] != (h, w):
                    hm = cv2.resize(hm, (w, h))
                hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
                output = cv2.addWeighted(output, 0.7, hm_color, 0.3, 0)
        
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
    
    def check_zone_breach(self, persons, motion_regions):
        """Check jika ada zone breach."""
        if self.zone_manager.get_zone_count() == 0:
            return False, ""
        
        for person in persons:
            x1, y1, x2, y2 = person.bbox
            check_points = [
                (x1, y1), (x2, y1), (x1, y2), (x2, y2),
                person.center, person.foot_center,
                ((x1 + x2) // 2, y1), ((x1 + x2) // 2, y2),
                (x1, (y1 + y2) // 2), (x2, (y1 + y2) // 2),
            ]
            
            for px, py in check_points:
                if self.zone_manager.check_all_zones(px, py):
                    return True, "Person detected in zone"
        
        return False, ""
    
    def toggle_arm(self, armed: bool):
        """Toggle arm/disarm."""
        self.is_armed = armed
        print(f"[System] Armed: {armed}")
        
        # Log ke database
        if armed:
            self.db.log_event(AlertType.SYSTEM.value, "System armed")
        else:
            self.db.log_event(AlertType.SYSTEM.value, "System disarmed")
    
    def toggle_record(self, recording: bool):
        """Toggle recording."""
        self.is_recording = recording
        
        if recording:
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = str(self.config.RECORDINGS_DIR / f"rec_{ts}.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(path, fourcc, 25.0, (1280, 720))
            print(f"[Record] Started: {path}")
        else:
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            print("[Record] Stopped")
    
    def toggle_mute(self, muted: bool):
        """Toggle mute."""
        self.is_muted = muted
        print(f"[Audio] Muted: {muted}")
    
    def take_snapshot(self) -> str:
        """Take snapshot."""
        with self.frame_lock:
            if self.current_frame is None:
                return None
            frame = self.current_frame.copy()
        
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = str(self.config.SNAPSHOTS_DIR / f"snap_{ts}.jpg")
        cv2.imwrite(path, frame)
        print(f"[Snapshot] Saved: {path}")
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return base64.b64encode(buffer).decode('utf-8')
    
    def set_confidence(self, conf: float):
        """Set confidence level."""
        self.confidence = conf
        if self.person_detector:
            self.person_detector.set_confidence(conf)
        print(f"[Detection] Confidence: {conf}")
    
    def set_model(self, model_name: str):
        """Set YOLO model."""
        self.model_name = model_name
        if self.person_detector:
            self.person_detector.change_model(model_name)
        print(f"[Detection] Model: {model_name}")
    
    def toggle_skeleton(self, enabled: bool):
        """Toggle skeleton detection."""
        self.enable_skeleton = enabled
        if self.detection_thread:
            self.detection_thread.draw_skeleton = enabled
        print(f"[Detection] Skeleton: {enabled}")
    
    def reload_faces(self):
        """Reload trusted faces."""
        if self.face_engine:
            self.face_engine.reload_faces()
        print(f"[Faces] Reloaded: {len(self.face_engine.known_names) if self.face_engine else 0}")
    
    def create_zone(self):
        """Create new zone."""
        self.zone_manager.create_zone()
        print(f"[Zone] Created: Total {self.zone_manager.get_zone_count()}")
    
    def add_zone_point(self, x: int, y: int):
        """Add point to active zone."""
        zone = self.zone_manager.get_active_zone()
        if zone:
            zone.add_point(x, y)
            print(f"[Zone] Point added: ({x}, {y})")
    
    def clear_zones(self):
        """Clear all zones."""
        self.zone_manager.delete_all_zones()
        self.breach_active = False
        print("[Zone] Cleared all zones")
    
    def stop(self):
        """Stop system."""
        self.running = False
        if self.cap:
            self.cap.release()
        if self.detection_thread:
            self.detection_thread.stop()
        if self.video_writer:
            self.video_writer.release()
        print("[System] Stopped")


class SecurityWebServer:
    """WebSocket server untuk web interface."""
    
    def __init__(self):
        self.system = SecurityWebSystem()
        self.clients: Set = set()
        self.running = False
        self.frame = None
        
    async def handle_client(self, websocket, path):
        """Handle WebSocket connection."""
        client_addr = websocket.remote_address
        print(f"[WebSocket] Client connected: {client_addr}")
        self.clients.add(websocket)
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_command(data, websocket)
                except Exception as e:
                    print(f"[WebSocket] Error handling message: {e}")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            print(f"[WebSocket] Client disconnected: {client_addr}")
    
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
                'zones': self.system.zone_manager.get_zone_count(),
                'faces': len(self.system.face_engine.known_names) if self.system.face_engine else 0,
                'clients': len(self.clients)
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
            print(f"[Detection] Face: {self.system.enable_face}")
        
        elif cmd_type == 'toggle_motion':
            self.system.enable_motion = data.get('value', True)
            print(f"[Detection] Motion: {self.system.enable_motion}")
        
        elif cmd_type == 'toggle_heatmap':
            self.system.enable_heatmap = data.get('value', False)
            print(f"[Detection] Heatmap: {self.system.enable_heatmap}")
        
        elif cmd_type == 'toggle_night_vision':
            self.system.night_vision = data.get('value', False)
            print(f"[Display] Night Vision: {self.system.night_vision}")
        
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
    
    async def start(self):
        """Start server."""
        print("[Server] Starting Security System...")
        
        # Start camera
        if not self.system.start_camera():
            print("[Server] Error: No camera detected")
            return False
        
        # Start detection
        if not self.system.start_detection():
            print("[Server] Error: Detection modules failed to load")
            return False
        
        self.system.running = True
        
        # Start frame capture loop
        async def capture_loop():
            while self.system.running:
                frame = self.system.capture_frame()
                if frame is not None:
                    processed = self.system.process_frame(frame)
                    await self.broadcast_frame(processed)
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
            print("[Server] WebSocket server started on ws://0.0.0.0:8765")
            print("[Server] Access web interface at: http://YOUR_SERVER_IP:8080")
            
            # Keep running
            while self.system.running:
                await asyncio.sleep(1)
        
        return True
    
    def stop(self):
        """Stop server."""
        self.system.stop()


async def main():
    server = SecurityWebServer()
    
    try:
        await server.start()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
        server.stop()


if __name__ == "__main__":
    asyncio.run(main())
