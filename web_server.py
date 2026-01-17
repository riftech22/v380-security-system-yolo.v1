#!/usr/bin/env python3
"""Web Server untuk Security System - Multi-threaded Version (Like Frigate)."""

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
import queue
from typing import Set, Dict, Optional
from collections import deque

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Import modules with error handling
try:
    from config import Config, AlertType
    from detectors import PersonDetector, FaceRecognitionEngine, MotionDetector, DetectionThread
    from database import DatabaseManager
    from utils import MultiZoneManager
    MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"[Import] Running in demo mode: {e}")
    MODULES_AVAILABLE = False

# Import V380 FFmpeg Pipeline
try:
    from v380_ffmpeg_pipeline import V380FFmpegProcessor
    V380_AVAILABLE = True
except ImportError as e:
    logging.warning(f"[Import] V380 FFmpeg Pipeline not available: {e}")
    V380_AVAILABLE = False


class FrameBuffer:
    """Thread-safe frame buffer pool."""
    
    def __init__(self, max_size: int = 3):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.max_size = max_size
    
    def put(self, frame: np.ndarray):
        """Add frame to buffer (non-blocking)."""
        with self.lock:
            if len(self.buffer) >= self.max_size:
                # Remove oldest frame
                self.buffer.popleft()
            self.buffer.append(frame.copy())
    
    def get(self) -> Optional[np.ndarray]:
        """Get latest frame (non-blocking)."""
        with self.lock:
            if self.buffer:
                return self.buffer[-1].copy()
        return None
    
    def get_all(self) -> list:
        """Get all frames from buffer."""
        with self.lock:
            return [frame.copy() for frame in self.buffer]
    
    def clear(self):
        """Clear buffer."""
        with self.lock:
            self.buffer.clear()


class CaptureThread:
    """Separate thread for frame capture from camera."""
    
    def __init__(self, camera_source, buffer_size: int = 3):
        self.camera_source = camera_source
        self.frame_buffer = FrameBuffer(max_size=buffer_size)
        self.running = False
        self.cap = None
        self.thread = None
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
    
    def start(self):
        """Start capture thread."""
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logging.info(f"[Capture] Thread started")
    
    def _capture_loop(self):
        """Capture loop - runs continuously."""
        try:
            self.cap = cv2.VideoCapture(self.camera_source)
            
            if not self.cap.isOpened():
                logging.error(f"[Capture] Failed to open camera: {self.camera_source}")
                return
            
            # Optimize RTSP settings
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Buffer size
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
            logging.info(f"[Capture] Connected to camera: {self.camera_source}")
            
            while self.running:
                try:
                    ret, frame = self.cap.read()
                    
                    if ret and frame is not None and isinstance(frame, np.ndarray):
                        # Force BGR conversion
                        if len(frame.shape) == 2 or frame.shape[2] == 1:
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        
                        # Validate frame
                        if frame.shape[0] > 0 and frame.shape[1] > 0:
                            self.frame_buffer.put(frame)
                            
                            # Calculate FPS
                            self.frame_count += 1
                            elapsed = time.time() - self.last_time
                            if elapsed >= 2.0:
                                self.fps = self.frame_count / elapsed
                                self.frame_count = 0
                                self.last_time = time.time()
                                logging.info(f"[Capture] FPS: {self.fps:.1f}, Buffer: {len(self.frame_buffer.buffer)}")
                    else:
                        logging.warning(f"[Capture] Failed to read frame")
                        time.sleep(0.01)
                
                except Exception as e:
                    logging.error(f"[Capture] Error: {e}")
                    time.sleep(0.01)
        
        except Exception as e:
            logging.error(f"[Capture] Fatal error: {e}")
        finally:
            if self.cap:
                self.cap.release()
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest frame (non-blocking)."""
        return self.frame_buffer.get()
    
    def stop(self):
        """Stop capture thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        logging.info("[Capture] Thread stopped")


class ProcessingThread:
    """Separate thread for detection and overlays."""
    
    def __init__(self, system):
        self.system = system
        self.running = False
        self.thread = None
        self.processed_frame = None
        self.lock = threading.Lock()
        self.frame_skip = 0  # Skip frames to reduce load
    
    def start(self):
        """Start processing thread."""
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        logging.info("[Processing] Thread started")
    
    def _process_loop(self):
        """Processing loop - runs continuously with frame skipping."""
        while self.running:
            try:
                frame = self.system.capture_thread.get_frame()
                
                if frame is not None:
                    # Frame skipping to reduce CPU load (process every 3rd frame)
                    self.frame_skip = (self.frame_skip + 1) % 3
                    
                    if self.frame_skip == 0:
                        # Submit frame to detection thread only on non-skipped frames
                        if self.system.detection_thread:
                            self.system.detection_thread.submit(frame)
                    
                    # Always process frame (even if skipping detection)
                    processed = self.system._process_frame_internal(frame)
                    
                    with self.lock:
                        self.processed_frame = processed
                
                # Small sleep to prevent CPU overload
                time.sleep(0.005)
            
            except Exception as e:
                logging.error(f"[Processing] Error: {e}")
                time.sleep(0.01)
    
    def get_processed_frame(self) -> Optional[np.ndarray]:
        """Get processed frame (non-blocking)."""
        with self.lock:
            if self.processed_frame is not None:
                return self.processed_frame.copy()
        return None
    
    def stop(self):
        """Stop processing thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logging.info("[Processing] Thread stopped")


class SecurityWebSystem:
    """Backend Security System dengan multi-threading."""
    
    def __init__(self, use_v380_ffmpeg: bool = False):
        self.config = None
        self.db = None
        self.zone_manager = None
        
        # Detectors
        self.person_detector = None
        self.face_engine = None
        self.motion_detector = None
        self.detection_thread = None
        
        # Multi-threading components
        self.capture_thread = None
        self.processing_thread = None
        
        # V380 FFmpeg Pipeline (Frigate-style)
        self.v380_processor = None
        self.use_v380_ffmpeg = use_v380_ffmpeg and V380_AVAILABLE
        
        # State
        self.running = False
        self.camera_available = False
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
        
        # Demo mode
        self.demo_mode = not MODULES_AVAILABLE
        
        # Video recording
        self.video_writer = None
        
        if not self.demo_mode:
            try:
                self.config = Config()
                self.db = DatabaseManager(self.config)
                self.zone_manager = MultiZoneManager()
            except Exception as e:
                logging.error(f"[Init] Error: {e}")
                self.demo_mode = True
    
    def start_camera(self):
        """Start camera dengan multi-threading."""
        logging.info("[Camera] Initializing...")
        
        # Check if using V380 FFmpeg Pipeline
        if self.use_v380_ffmpeg:
            logging.info("[Camera] Using V380 FFmpeg Pipeline (Frigate-style)")
            
            rtsp_url = Config.CAMERA_SOURCE if self.config else "rtsp://admin:admin@192.168.1.108:554/live"
            
            try:
                self.v380_processor = V380FFmpegProcessor(
                    rtsp_url=rtsp_url,
                    model_path="yolov8s.pt",
                    detect_fps=5,
                    device="cpu",
                    conf_threshold=self.confidence,
                    iou_threshold=0.45
                )
                self.v380_processor.start()
                
                # Wait for initial frames
                time.sleep(3)
                
                # Check if we have detections
                result = self.v380_processor.get_latest_detection()
                if result:
                    self.camera_available = True
                    logging.info("[Camera] V380 FFmpeg Pipeline connected successfully")
                else:
                    logging.warning("[Camera] No frames from V380, falling back to demo mode")
                    self.camera_available = False
                    
            except Exception as e:
                logging.error(f"[Camera] V380 FFmpeg Pipeline error: {e}")
                self.camera_available = False
        else:
            # Use original OpenCV capture
            camera_source = Config.CAMERA_SOURCE if self.config else 0
            
            # Start capture thread
            self.capture_thread = CaptureThread(camera_source, buffer_size=3)
            self.capture_thread.start()
            
            # Wait for first frame
            time.sleep(1.0)
            
            if self.capture_thread.get_frame() is not None:
                self.camera_available = True
                logging.info("[Camera] Connected successfully (OpenCV)")
            else:
                logging.warning("[Camera] No frame detected, using demo mode")
                self.camera_available = False
    
    def start_detection(self):
        """Start detection modules."""
        if self.demo_mode:
            logging.info("[Detection] Demo mode - skipping detection")
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
            return True  # Continue without detection
    
    def start_processing(self):
        """Start processing thread."""
        self.processing_thread = ProcessingThread(self)
        self.processing_thread.start()
        logging.info("[Processing] Thread started")
    
    def _get_v380_frame(self) -> Optional[np.ndarray]:
        """Get processed frame from V380 FFmpeg Pipeline."""
        if not self.v380_processor:
            return None
        
        try:
            result = self.v380_processor.get_latest_detection()
            if not result:
                return None
            
            # Draw detections on both splits
            top_frame_drawn = self.v380_processor.draw_detections(
                result['top_frame'],
                result['top_detections'],
                "Top Camera (Fixed)"
            )
            
            bottom_frame_drawn = self.v380_processor.draw_detections(
                result['bottom_frame'],
                result['bottom_detections'],
                "Bottom Camera (PTZ)"
            )
            
            # Stack frames vertically
            combined_frame = np.vstack([top_frame_drawn, bottom_frame_drawn])
            
            # Add separator line
            h = combined_frame.shape[0] // 2
            cv2.line(combined_frame, (0, h), (combined_frame.shape[1], h), (0, 255, 255), 2)
            
            # Update person count
            top_count = len(result['top_detections'].boxes) if result['top_detections'] else 0
            bottom_count = len(result['bottom_detections'].boxes) if result['bottom_detections'] else 0
            self.person_count = top_count + bottom_count
            
            return combined_frame
            
        except Exception as e:
            logging.error(f"[V380] Error getting frame: {e}")
            return None
    
    def _process_frame_internal(self, frame: np.ndarray) -> np.ndarray:
        """Internal frame processing dengan semua fitur - FIXED frame corruption."""
        try:
            # Check if using V380 FFmpeg Pipeline
            if self.use_v380_ffmpeg:
                v380_frame = self._get_v380_frame()
                if v380_frame is not None:
                    # Add timestamp and status overlay to V380 frame
                    h, w = v380_frame.shape[:2]
                    
                    # Timestamp
                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(v380_frame, ts, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # V380 mode indicator
                    cv2.putText(v380_frame, "V380 SPLIT MODE", (10, h - 15), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    # Recording indicator
                    if self.is_recording:
                        cv2.circle(v380_frame, (w - 25, 25), 10, (0, 0, 255), -1)
                        cv2.putText(v380_frame, "REC", (w - 65, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Armed indicator
                    if self.is_armed:
                        cv2.putText(v380_frame, "ARMED", (w - 85, h - 45), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # FPS info
                    fps_text = f"Capture: {self.v380_processor.capture_fps} | Detect: {self.v380_processor.detection_fps}"
                    cv2.putText(v380_frame, fps_text, (10, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    return v380_frame
            
            # Validate input frame
            if frame is None or frame.size == 0:
                return self._create_demo_frame()
            
            # Ensure frame is valid numpy array
            if not isinstance(frame, np.ndarray):
                return self._create_demo_frame()
            
            # Ensure correct shape and type
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                # Convert grayscale or invalid format to BGR
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                else:
                    return self._create_demo_frame()
            
            # Ensure uint8 type
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            # Get dimensions
            h, w = frame.shape[:2]
            
            # Validate frame dimensions
            if h == 0 or w == 0:
                return self._create_demo_frame()
            
            # Resize only if necessary (avoid unnecessary processing)
            target_h, target_w = 720, 1280
            if h != target_h or w != target_w:
                try:
                    frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    h, w = target_h, target_w
                except Exception as e:
                    logging.error(f"[Resize] Error: {e}")
                    return self._create_demo_frame()
            
            # Create output copy
            output = frame.copy()
            
            # Validate output
            if output is None or output.size == 0:
                return self._create_demo_frame()
            
            # Timestamp
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(output, ts, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Get detection results (non-blocking)
            if not self.demo_mode and self.detection_thread:
                try:
                    results = self.detection_thread.get_results()
                    persons = results.get('persons', []) if results else []
                    has_motion = results.get('motion', False) if results else False
                    motion_regions = results.get('motion_regions', []) if results else []
                except Exception as e:
                    logging.error(f"[Detection] Error getting results: {e}")
                    persons = []
                    has_motion = False
                    motion_regions = []
            else:
                persons = []
                has_motion = False
                motion_regions = []
            
            self.person_count = len(persons)
            
            # Night vision (safe processing)
            if self.night_vision:
                try:
                    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
                    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    output[:, :, 1] = np.clip(output[:, :, 1] * 1.3 + 30, 0, 255).astype(np.uint8)
                except Exception as e:
                    logging.error(f"[NightVision] Error: {e}")
            
            # Heat map (safe processing)
            if self.enable_heatmap and self.motion_detector and not self.demo_mode:
                try:
                    hm = self.motion_detector.get_heat_map()
                    if hm is not None and hm.size > 0:
                        if hm.shape[:2] != (h, w):
                            hm = cv2.resize(hm, (w, h))
                        hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
                        output = cv2.addWeighted(output, 0.7, hm_color, 0.3, 0)
                except Exception as e:
                    logging.error(f"[Heatmap] Error: {e}")
            
            # Motion boxes
            if self.enable_motion and motion_regions:
                try:
                    for mx1, my1, mx2, my2 in motion_regions:
                        # Validate coordinates
                        if mx1 >= 0 and my1 >= 0 and mx2 <= w and my2 <= h:
                            cv2.rectangle(output, (mx1, my1), (mx2, my2), (0, 165, 255), 1)
                except Exception as e:
                    logging.error(f"[MotionBoxes] Error: {e}")
            
            # Draw persons dengan bounding boxes
            if persons:
                try:
                    for person in persons:
                        x1, y1, x2, y2 = person.bbox
                        # Validate coordinates
                        if x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h:
                            color = (0, 0, 255) if self.breach_active else (0, 255, 0)
                            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(output, f"Person {person.confidence:.0%}", 
                                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                except Exception as e:
                    logging.error(f"[PersonBoxes] Error: {e}")
            
            # Draw zones
            if self.zone_manager:
                try:
                    breached_ids = []
                    if self.breach_active:
                        for zone in self.zone_manager.zones:
                            breached_ids.append(zone.zone_id)
                    output = self.zone_manager.draw_all(output, breached_ids, self.is_armed)
                except Exception as e:
                    logging.error(f"[Zones] Error: {e}")
            
            # Recording indicator
            if self.is_recording:
                cv2.circle(output, (w - 25, 25), 10, (0, 0, 255), -1)
                cv2.putText(output, "REC", (w - 65, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Armed indicator
            if self.is_armed:
                cv2.putText(output, "ARMED", (w - 85, h - 15), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Final validation
            if output is None or output.size == 0:
                return self._create_demo_frame()
            
            return output
            
        except Exception as e:
            logging.error(f"[ProcessFrame] Fatal error: {e}")
            return self._create_demo_frame()
    
    def _create_demo_frame(self):
        """Create demo frame."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:, :] = (30, 30, 40)
        
        cv2.putText(frame, "RIFTECH CAM SECURITY", (340, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.putText(frame, "Demo Mode - No Camera", (320, 260),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 200), 2)
        
        t = time.time()
        x = int(640 + 200 * np.sin(t))
        y = int(360 + 100 * np.cos(t))
        cv2.circle(frame, (x, y), 20, (0, 255, 0), -1)
        
        cv2.putText(frame, f"Server Time: {time.strftime('%H:%M:%S')}", (400, 450),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    # Control methods
    def toggle_arm(self, armed: bool):
        self.is_armed = armed
        logging.info(f"[System] Armed: {armed}")
    
    def toggle_record(self, recording: bool):
        self.is_recording = recording
        logging.info(f"[Record] Recording: {recording}")
    
    def toggle_mute(self, muted: bool):
        self.is_muted = muted
        logging.info(f"[Audio] Muted: {muted}")
    
    def take_snapshot(self) -> str:
        frame = self.processing_thread.get_processed_frame()
        if frame is None:
            frame = self._create_demo_frame()
        
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = f"snapshots/snap_{ts}.jpg"
        try:
            cv2.imwrite(path, frame)
            logging.info(f"[Snapshot] Saved: {path}")
        except Exception as e:
            logging.error(f"[Snapshot] Error: {e}")
        
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return base64.b64encode(buffer).decode('utf-8')
    
    def set_confidence(self, conf: float):
        self.confidence = conf
        if self.person_detector:
            self.person_detector.set_confidence(conf)
    
    def set_model(self, model_name: str):
        self.model_name = model_name
        if self.person_detector:
            self.person_detector.change_model(model_name)
    
    def toggle_skeleton(self, enabled: bool):
        self.enable_skeleton = enabled
        if self.detection_thread:
            self.detection_thread.draw_skeleton = enabled
    
    def reload_faces(self):
        if self.face_engine:
            self.face_engine.reload_faces()
    
    def create_zone(self):
        if self.zone_manager:
            self.zone_manager.create_zone()
    
    def add_zone_point(self, x: int, y: int):
        if self.zone_manager:
            zone = self.zone_manager.get_active_zone()
            if zone:
                zone.add_point(x, y)
    
    def clear_zones(self):
        if self.zone_manager:
            self.zone_manager.delete_all_zones()
        self.breach_active = False
    
    def stop(self):
        """Stop system."""
        self.running = False
        if self.capture_thread:
            self.capture_thread.stop()
        if self.processing_thread:
            self.processing_thread.stop()
        if self.detection_thread:
            self.detection_thread.stop()
        if self.video_writer:
            self.video_writer.release()


class SecurityWebServer:
    """WebSocket server dengan non-blocking broadcast."""
    
    def __init__(self, use_v380_ffmpeg: bool = False):
        self.system = SecurityWebSystem(use_v380_ffmpeg=use_v380_ffmpeg)
        self.clients: Set = set()
        self.running = False
        self.broadcast_queue = queue.Queue(maxsize=10)
    
    async def handle_client(self, websocket):
        client_addr = websocket.remote_address
        logging.info(f"[WebSocket] Client connected: {client_addr}")
        self.clients.add(websocket)
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_command(data, websocket)
                except Exception as e:
                    logging.error(f"[WebSocket] Error: {e}")
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logging.error(f"[WebSocket] Error: {e}")
        finally:
            self.clients.discard(websocket)
            logging.info(f"[WebSocket] Client disconnected: {client_addr}")
    
    async def handle_command(self, data: Dict, websocket):
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
        
        elif cmd_type == 'toggle_motion':
            self.system.enable_motion = data.get('value', True)
        
        elif cmd_type == 'toggle_heatmap':
            self.system.enable_heatmap = data.get('value', False)
        
        elif cmd_type == 'toggle_night_vision':
            self.system.night_vision = data.get('value', False)
        
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
    
    async def broadcast_task(self):
        """Non-blocking broadcast task - Fix frame stacking with validation."""
        frame_count = 0
        last_log_time = time.time()
        target_fps = 15
        frame_interval = 1.0 / target_fps
        
        while self.system.running:
            try:
                start_time = time.time()
                
                # Get processed frame from processing thread
                frame = self.system.processing_thread.get_processed_frame()
                
                if frame is not None:
                    # Validate frame before encoding
                    if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
                        logging.warning("[Broadcast] Invalid frame, skipping")
                        await asyncio.sleep(frame_interval)
                        continue
                    
                    # Ensure BGR format
                    if len(frame.shape) == 2 or frame.shape[2] == 1:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    
                    # Optimized encoding with error handling
                    try:
                        encode_params = [
                            cv2.IMWRITE_JPEG_QUALITY, 65,
                            cv2.IMWRITE_JPEG_OPTIMIZE, 0,
                            cv2.IMWRITE_JPEG_PROGRESSIVE, 0
                        ]
                        
                        success, buffer = cv2.imencode('.jpg', frame, encode_params)
                        if not success:
                            logging.warning("[Broadcast] Encoding failed, skipping frame")
                            await asyncio.sleep(frame_interval)
                            continue
                        
                        frame_b64 = base64.b64encode(buffer).decode('utf-8')
                        
                        message = json.dumps({
                            'type': 'frame',
                            'timestamp': time.time(),
                            'data': frame_b64
                        })
                        
                        # Non-blocking broadcast
                        if self.clients:
                            try:
                                await asyncio.gather(
                                    *[client.send(message) for client in self.clients],
                                    return_exceptions=True
                                )
                            except Exception as e:
                                logging.warning(f"[Broadcast] Send error: {e}")
                        
                        frame_count += 1
                    
                    except Exception as e:
                        logging.error(f"[Broadcast] Encoding error: {e}")
                        await asyncio.sleep(frame_interval)
                        continue
                
                # Log FPS every 2 seconds
                if time.time() - last_log_time >= 2.0:
                    actual_fps = frame_count / (time.time() - last_log_time)
                    logging.info(f"[Broadcast] FPS: {actual_fps:.1f}, Clients: {len(self.clients)}")
                    frame_count = 0
                    last_log_time = time.time()
                
                # Calculate sleep time
                elapsed = time.time() - start_time
                sleep_time = max(0.01, frame_interval - elapsed)
                await asyncio.sleep(sleep_time)
            
            except Exception as e:
                logging.error(f"[Broadcast] Fatal error: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(0.05)
    
    async def start(self):
        """Start server dengan multi-threading."""
        logging.info("[Server] Starting Security System (Multi-threaded)...")
        
        try:
            # Start capture thread
            self.system.start_camera()
            
            # Start detection
            self.system.start_detection()
            
            # Start processing thread
            self.system.start_processing()
            
            self.system.running = True
            
            # Start broadcast task
            asyncio.create_task(self.broadcast_task())
            
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
                logging.info("[Server] Multi-threaded architecture:")
                logging.info("  - Capture Thread: Continuous frame capture")
                logging.info("  - Processing Thread: Detection & overlays")
                logging.info("  - Broadcast Task: Non-blocking WebSocket")
                
                if self.system.demo_mode:
                    logging.info("[Server] Running in DEMO MODE")
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
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Security Web Server with V380 Support')
    parser.add_argument('--v380', action='store_true',
                       help='Use V380 FFmpeg Pipeline for split frame processing')
    parser.add_argument('--rtsp', type=str, default=None,
                       help='RTSP URL for camera (default: from config)')
    args = parser.parse_args()
    
    # Start server
    if args.v380:
        logging.info("[Server] V380 FFmpeg Mode Enabled")
        if V380_AVAILABLE:
            logging.info("[Server] V380 FFmpeg Pipeline is available")
        else:
            logging.error("[Server] V380 FFmpeg Pipeline is NOT available")
            logging.error("[Server] Please ensure v380_ffmpeg_pipeline.py exists")
            sys.exit(1)
    
    server = SecurityWebServer(use_v380_ffmpeg=args.v380)
    
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
    logging.info("Starting Security Web Server (Multi-threaded)...")
    logging.info("[Server] Usage:")
    logging.info("  Normal mode:    python3 web_server.py")
    logging.info("  V380 mode:      python3 web_server.py --v380")
    logging.info("  Custom RTSP:     python3 web_server.py --v380 --rtsp rtsp://user:pass@ip:554/live")
    asyncio.run(main())
