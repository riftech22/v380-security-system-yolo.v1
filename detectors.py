#!/usr/bin/env python3
"""Detection modules - Professional skeleton, accurate zone detection."""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import threading
from queue import Queue, Empty
import shutil
import gc

from config import Config, Sensitivity


class DownloadManager:
    """Track model downloads."""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._downloading = False
                    cls._instance._model_name = ""
                    cls._instance._progress = 0
        return cls._instance
    
    def start_download(self, model_name: str):
        self._downloading = True
        self._model_name = model_name
    
    def end_download(self):
        self._downloading = False
        self._model_name = ""
    
    def get_status(self) -> Tuple[bool, str, int]:
        return self._downloading, self._model_name, self._progress


download_manager = DownloadManager()


YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    pass

FACE_RECOGNITION_AVAILABLE = False
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    pass

MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    pass


@dataclass
class SkeletonLandmark:
    x: int
    y: int
    visibility: float
    name: str


@dataclass 
class PersonDetection:
    center: Tuple[int, int]
    foot_center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    confidence: float
    skeleton_landmarks: List[SkeletonLandmark] = field(default_factory=list)
    track_id: int = -1


@dataclass
class FaceDetection:
    name: str
    confidence: float
    is_trusted: bool
    bbox: Tuple[int, int, int, int]


class PersonDetector:
    """Person detector with professional skeleton drawing."""
    
    # MediaPipe pose landmark indices
    NOSE = 0
    LEFT_EYE = 2
    RIGHT_EYE = 5
    LEFT_EAR = 7
    RIGHT_EAR = 8
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    
    # Professional skeleton connections with colors
    SKELETON_CONNECTIONS = [
        # Head (cyan)
        (NOSE, LEFT_EYE, (0, 255, 255)),
        (NOSE, RIGHT_EYE, (0, 255, 255)),
        (LEFT_EYE, LEFT_EAR, (0, 255, 255)),
        (RIGHT_EYE, RIGHT_EAR, (0, 255, 255)),
        
        # Torso (green)
        (LEFT_SHOULDER, RIGHT_SHOULDER, (0, 255, 0)),
        (LEFT_SHOULDER, LEFT_HIP, (0, 255, 0)),
        (RIGHT_SHOULDER, RIGHT_HIP, (0, 255, 0)),
        (LEFT_HIP, RIGHT_HIP, (0, 255, 0)),
        
        # Left arm (yellow)
        (LEFT_SHOULDER, LEFT_ELBOW, (0, 255, 255)),
        (LEFT_ELBOW, LEFT_WRIST, (0, 200, 255)),
        
        # Right arm (yellow)
        (RIGHT_SHOULDER, RIGHT_ELBOW, (0, 255, 255)),
        (RIGHT_ELBOW, RIGHT_WRIST, (0, 200, 255)),
        
        # Left leg (magenta)
        (LEFT_HIP, LEFT_KNEE, (255, 0, 255)),
        (LEFT_KNEE, LEFT_ANKLE, (255, 100, 255)),
        
        # Right leg (magenta)
        (RIGHT_HIP, RIGHT_KNEE, (255, 0, 255)),
        (RIGHT_KNEE, RIGHT_ANKLE, (255, 100, 255)),
    ]
    
    # Joint colors by body part
    JOINT_COLORS = {
        'head': (0, 255, 255),      # Cyan
        'torso': (0, 255, 0),       # Green
        'arm': (0, 200, 255),       # Orange-yellow
        'leg': (255, 100, 255),     # Pink-magenta
    }
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.model_name = 'yolov8n.pt'
        self.confidence = config.YOLO_CONFIDENCE
        self._lock = threading.Lock()
        self._loaded = False
        
        # MediaPipe pose for skeleton
        self.pose = None
        self.mp_pose = None
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
        
        if YOLO_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        try:
            download_manager.start_download(self.model_name)
            print(f"[Detector] Loading {self.model_name}...")
            self.model = YOLO(self.model_name)
            self._loaded = True
            print(f"[Detector] Loaded")
        except Exception as e:
            print(f"[Detector] Error: {e}")
        finally:
            download_manager.end_download()
    
    def _init_pose(self):
        """Lazy init MediaPipe pose."""
        if self.pose is None and MEDIAPIPE_AVAILABLE and self.mp_pose:
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
    
    def change_model(self, model_name: str):
        if not YOLO_AVAILABLE:
            return
        
        def load():
            download_manager.start_download(model_name)
            with self._lock:
                try:
                    self.model = YOLO(model_name)
                    self.model_name = model_name
                    self._loaded = True
                except MemoryError:
                    print("[Detector] Memory error")
                    gc.collect()
                except Exception as e:
                    print(f"[Detector] Error: {e}")
                finally:
                    download_manager.end_download()
        
        threading.Thread(target=load, daemon=True).start()
    
    def set_sensitivity(self, sensitivity: Sensitivity):
        settings = Config.get_sensitivity_settings(sensitivity)
        self.confidence = settings.get('yolo_confidence', 0.25)
    
    def set_confidence(self, conf: float):
        self.confidence = max(0.1, min(0.99, conf))
    
    def _enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Enhance frame for better detection using Frigate-style preprocessing.
        
        Args:
            frame: Input frame
            
        Returns:
            Enhanced frame
        """
        # Convert to LAB color space for better lighting adjustment
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge back and convert to BGR
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Slight brightness adjustment
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=10)
        
        return enhanced
    
    def _letterbox_resize(self, frame: np.ndarray, target_size: int = 640) -> np.ndarray:
        """Resize frame with letterboxing (maintain aspect ratio).
        
        Args:
            frame: Input frame
            target_size: Target size (square, e.g., 640x640)
            
        Returns:
            Resized frame with letterboxing
        """
        h, w = frame.shape[:2]
        
        # Calculate scale factor to fit within target_size
        scale = min(target_size / w, target_size / h)
        
        # Resize with aspect ratio maintained
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create letterbox (black padding)
        letterbox = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        
        # Center the resized frame
        x_offset = (target_size - new_w) // 2
        y_offset = (target_size - new_h) // 2
        letterbox[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return letterbox
    
    def _preprocess_frame(self, frame: np.ndarray, target_size: tuple = (1280, 720)) -> np.ndarray:
        """Preprocess frame for better detection.
        
        Args:
            frame: Input frame (can be any resolution)
            target_size: Target resolution (default: 1280x720)
            
        Returns:
            Preprocessed frame at target_size
        """
        h, w = frame.shape[:2]
        
        # Step 1: Resize to target resolution ONLY if different
        # But DON'T resize if frame is already 1280x360 (split frame!)
        if (w, h) != target_size and (w, h) != (1280, 360):
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Step 2: Calculate brightness statistics
        brightness = np.mean(frame)
        std_dev = np.std(frame)
        
        # Step 3: Brightness adjustment
        if brightness > 120:  # Overexposed
            frame = cv2.convertScaleAbs(frame, alpha=0.7, beta=-30)
        elif brightness < 50:  # Underexposed
            frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=30)
        
        # Step 4: CLAHE for better contrast
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        frame = cv2.merge([l, a, b])
        frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
        
        return frame
    
    def _apply_nms(self, detections, iou_threshold=0.45, conf_threshold=0.5):
        """Apply Non-Maximum Suppression with Frigate-style parameters.
        
        Args:
            detections: List of PersonDetection objects
            iou_threshold: IoU threshold for suppression (Frigate: 0.45)
            conf_threshold: Confidence threshold (Frigate: 0.5)
            
        Returns:
            Filtered detections
        """
        if len(detections) <= 1:
            return detections
        
        # Filter by confidence threshold first (Frigate min_score: 0.5)
        detections = [d for d in detections if d.confidence >= conf_threshold]
        
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence (highest first)
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        # Initialize list to keep
        keep = []
        
        while detections:
            # Pick detection with highest confidence
            current = detections[0]
            keep.append(current)
            detections.pop(0)
            
            # Calculate IoU with remaining detections
            remaining = []
            for det in detections:
                iou = self._calculate_iou(current.bbox, det.bbox)
                
                # Keep only if IoU is below threshold (Frigate: 0.45)
                if iou < iou_threshold:
                    remaining.append(det)
            
            detections = remaining
        
        return keep
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU) for two bounding boxes.
        
        Args:
            bbox1: (x1, y1, x2, y2)
            bbox2: (x1, y1, x2, y2)
            
        Returns:
            IoU value (0-1)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)
        
        # Check if there's overlap
        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        else:
            inter_area = 0
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        # Calculate IoU
        if union_area > 0:
            return inter_area / union_area
        else:
            return 0.0
    
    def detect_split_frame(self, frame, top_conf=0.4, bottom_conf=0.4):
        """Detect persons in V380 split frame with enhanced dual-lens handling for high resolution.
        
        V380 Dual-Lens Camera Anatomy (Dynamic Resolution Support):
        - Total Frame: Can be 1280x720 (HD) or 2304x2592 (2K) - vertical stack
        - Top Split (Fixed Wide): (0,0) to (width,height//2) - Wide angle, area monitoring
        - Bottom Split (PTZ): (0,height//2) to (width,height) - Tracking, detail view
        
        Key Considerations:
        1. Each split has extreme aspect ratio (e.g., 2304:1296 = 1.78:1 or 1280:360 = 3.5:1)
        2. High resolution (2K/4K) requires downscaling for performance
        3. Without proper letterboxing, people appear distorted causing AI confusion
        
        Args:
            frame: Input frame (any resolution, will be dynamically split)
            top_conf: Confidence threshold for top camera (wide angle)
            bottom_conf: Confidence threshold for bottom camera (PTZ tracking)
            
        Returns:
            List of PersonDetection objects with adjusted coordinates
        """
        # Step 1: Get frame dimensions
        h, w = frame.shape[:2]
        print(f"[V380 Split] Input frame: {w}x{h}")
        
        # Step 2: Dynamic split point at middle of frame
        # This works for ANY vertical stack resolution (1280x720, 2304x2592, etc.)
        split_point = h // 2
        print(f"[V380 Split] Split point at y={split_point}")
        
        # Step 3: Crop frame at split point
        top_frame_raw = frame[:split_point, :]  # Top lens: width x (height//2)
        bottom_frame_raw = frame[split_point:, :]  # Bottom lens: width x (height//2)
        print(f"[V380 Split] Top crop: {top_frame_raw.shape[1]}x{top_frame_raw.shape[0]}, Bottom crop: {bottom_frame_raw.shape[1]}x{bottom_frame_raw.shape[0]}")
        mid_y = split_point  # Keep mid_y for coordinate mapping to full frame
        
        # Step 4: Preprocess each crop separately
        top_frame = self._preprocess_frame(top_frame_raw)
        bottom_frame = self._preprocess_frame(bottom_frame_raw)
        print(f"[V380 Split] Preprocessed top: {top_frame.shape[1]}x{top_frame.shape[0]}, bottom: {bottom_frame.shape[1]}x{bottom_frame.shape[0]}")
        
        # Step 5: Resize with LETTERBOXING (maintain aspect ratio, don't stretch!)
        # This prevents distortion of human shapes due to extreme aspect ratio (3.5:1)
        DETECTION_SIZE = 640
        top_frame_detect = self._letterbox_resize(top_frame, DETECTION_SIZE)
        bottom_frame_detect = self._letterbox_resize(bottom_frame, DETECTION_SIZE)
        print(f"[V380 Split] Letterboxed to: {top_frame_detect.shape[1]}x{top_frame_detect.shape[0]}")
        
        top_persons = []
        bottom_persons = []
        
        # Frigate-style filters (from objects.py FilterConfig)
        min_area = 5000       # Increased minimum area (reject tiny detections)
        max_area = 50000      # Much smaller maximum area (reject large bboxes)
        min_ratio = 0.4      # Person aspect ratio (width/height) - more strict
        max_ratio = 1.3       # Tighter max ratio (reject wide rectangles)
        
        # Detect in top camera (already enhanced by preprocessing)
        if top_frame_detect.size > 0:
            try:
                with self._lock:
                    results_top = self.model(top_frame_detect, conf=top_conf, classes=[0], verbose=False)
                
                if results_top and results_top[0].boxes:
                    for box in results_top[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf = float(box.conf[0])
                        
                        # Scale coordinates back from letterbox (640x640 -> top_frame size)
                        top_h, top_w = top_frame.shape[:2]
                        
                        # Calculate letterbox offsets
                        scale = min(DETECTION_SIZE / top_w, DETECTION_SIZE / top_h)
                        new_w = int(top_w * scale)
                        new_h = int(top_h * scale)
                        x_offset = (DETECTION_SIZE - new_w) // 2
                        y_offset = (DETECTION_SIZE - new_h) // 2
                        
                        # Remove letterbox offset and scale back to original size
                        x1 = int((x1 - x_offset) / scale)
                        y1 = int((y1 - y_offset) / scale)
                        x2 = int((x2 - x_offset) / scale)
                        y2 = int((y2 - y_offset) / scale)
                        
                        # Calculate area
                        area = (x2 - x1) * (y2 - y1)
                        
                        # Additional filter: Reject bounding boxes that are too large (near full frame)
                        top_area = top_frame.shape[0] * top_frame.shape[1]
                        if area > top_area * 0.15:  # Reject if bbox > 15% of frame (MUCH STRICTER)
                            print(f"[Split-Top] REJECT: area={area} too large (>{top_area * 0.15:.0f} = {area/top_area*100:.1f}%)")
                            continue
                        
                        # Frigate-style area filtering
                        if area < min_area or area > max_area:
                            print(f"[Split-Top] REJECT: area={area} out of range [{min_area}, {max_area}]")
                            continue
                        
                        # Calculate aspect ratio (width/height)
                        width = x2 - x1
                        height = y2 - y1
                        aspect_ratio = width / height if height > 0 else 0
                        
                        # Frigate-style aspect ratio filtering (person: 0.4-1.3) - MORE STRICT
                        if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
                            print(f"[Split-Top] REJECT: aspect_ratio={aspect_ratio:.2f} out of range [{min_ratio}, {max_ratio}]")
                            continue
                        
                        # IMPORTANT: Constraint bounding box to stay within top split (0 to 360)
                        # Prevent bbox from crossing split boundary
                        y1 = max(0, y1)
                        y2 = min(split_point - 5, y2)  # Leave 5px margin from split line
                        
                        # Ensure bbox is valid after constraints
                        if y2 <= y1 or (x2 - x1) < 10 or (y2 - y1) < 10:
                            print(f"[Split-Top] REJECT: Invalid bbox after constraints")
                            continue
                        
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        
                        person = PersonDetection(
                            center=(cx, cy),
                            foot_center=(cx, y2),
                            bbox=(x1, y1, x2, y2),
                            confidence=conf,
                            track_id=-1
                        )
                        top_persons.append(person)
            except Exception as e:
                print(f"[Split-Top] Error: {e}")
        
        # Detect in bottom camera (already enhanced by preprocessing)
        if bottom_frame_detect.size > 0:
            try:
                with self._lock:
                    results_bottom = self.model(bottom_frame_detect, conf=bottom_conf, classes=[0], verbose=False)
                
                if results_bottom and results_bottom[0].boxes:
                    for box in results_bottom[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf = float(box.conf[0])
                        
                        # Scale coordinates back from letterbox (640x640 -> bottom_frame size)
                        bottom_h, bottom_w = bottom_frame.shape[:2]
                        
                        # Calculate letterbox offsets
                        scale = min(DETECTION_SIZE / bottom_w, DETECTION_SIZE / bottom_h)
                        new_w = int(bottom_w * scale)
                        new_h = int(bottom_h * scale)
                        x_offset = (DETECTION_SIZE - new_w) // 2
                        y_offset = (DETECTION_SIZE - new_h) // 2
                        
                        # Remove letterbox offset and scale back to original size
                        x1 = int((x1 - x_offset) / scale)
                        y1 = int((y1 - y_offset) / scale)
                        x2 = int((x2 - x_offset) / scale)
                        y2 = int((y2 - y_offset) / scale)
                        
                        # Calculate area
                        area = (x2 - x1) * (y2 - y1)
                        
                        # Additional filter: Reject bounding boxes that are too large (near full frame)
                        bottom_area = bottom_frame.shape[0] * bottom_frame.shape[1]
                        if area > bottom_area * 0.15:  # Reject if bbox > 15% of frame (MUCH STRICTER)
                            print(f"[Split-Bottom] REJECT: area={area} too large (>{bottom_area * 0.15:.0f} = {area/bottom_area*100:.1f}%)")
                            continue
                        
                        # Frigate-style area filtering
                        if area < min_area or area > max_area:
                            print(f"[Split-Bottom] REJECT: area={area} out of range [{min_area}, {max_area}]")
                            continue
                        
                        # Calculate aspect ratio (width/height)
                        width = x2 - x1
                        height = y2 - y1
                        aspect_ratio = width / height if height > 0 else 0
                        
                        # Frigate-style aspect ratio filtering (person: 0.4-1.3) - MORE STRICT
                        if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
                            print(f"[Split-Bottom] REJECT: aspect_ratio={aspect_ratio:.2f} out of range [{min_ratio}, {max_ratio}]")
                            continue
                        
                        # IMPORTANT: Constraint bounding box to stay within bottom split (360 to 720)
                        # Prevent bbox from crossing split boundary
                        y1 = max(5, y1)  # Leave 5px margin from split line
                        y2 = min(bottom_h, y2)
                        
                        # Ensure bbox is valid after constraints
                        if y2 <= y1 or (x2 - x1) < 10 or (y2 - y1) < 10:
                            print(f"[Split-Bottom] REJECT: Invalid bbox after constraints")
                            continue
                        
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        
                        # Adjust coordinates to full frame (bottom region: mid_y to h)
                        person = PersonDetection(
                            center=(cx, cy + mid_y),
                            foot_center=(cx, y2 + mid_y),
                            bbox=(x1, y1 + mid_y, x2, y2 + mid_y),
                            confidence=conf,
                            track_id=-1
                        )
                        bottom_persons.append(person)
            except Exception as e:
                print(f"[Split-Bottom] Error: {e}")
        
        # Apply Frigate-style NMS (IoU=0.45, conf=0.5)
        top_persons = self._apply_nms(top_persons, iou_threshold=0.45, conf_threshold=0.5)
        bottom_persons = self._apply_nms(bottom_persons, iou_threshold=0.45, conf_threshold=0.5)
        
        # Refine bounding boxes using skeleton keypoints for tighter fit
        if MEDIAPIPE_AVAILABLE:
            self._init_pose()
            if self.pose:
                # Use preprocessed frames for skeleton refinement
                top_persons = self._refine_bboxes_with_skeleton(top_frame, top_persons, offset_y=0)
                bottom_persons = self._refine_bboxes_with_skeleton(bottom_frame, bottom_persons, offset_y=mid_y)
        
        # Merge and assign track IDs
        all_persons = []
        track_id = 0
        
        for person in top_persons:
            person.track_id = track_id
            all_persons.append(person)
            track_id += 1
        
        for person in bottom_persons:
            person.track_id = track_id
            all_persons.append(person)
            track_id += 1
        
        print(f"[V380 Split] Final result: {len(all_persons)} persons detected (Top: {len(top_persons)}, Bottom: {len(bottom_persons)})")
        return all_persons
    
    def _refine_bboxes_with_skeleton(self, frame: np.ndarray, persons: List[PersonDetection], offset_y: int = 0) -> List[PersonDetection]:
        """Refine bounding boxes using MediaPipe skeleton keypoints for tighter fit.
        
        Args:
            frame: Full processed frame
            persons: List of PersonDetection objects to refine
            offset_y: Y offset for split frame detection
            
        Returns:
            List of PersonDetection with refined bounding boxes
        """
        if not self.pose or not persons:
            return persons
        
        refined_persons = []
        
        for person in persons:
            x1, y1, x2, y2 = person.bbox
            
            # Crop person region for skeleton detection
            pad_x = int((x2 - x1) * 0.2)
            pad_y = int((y2 - y1) * 0.2)
            
            crop_x1 = max(0, x1 - pad_x)
            crop_y1 = max(0, y1 - pad_y)
            crop_x2 = min(frame.shape[1], x2 + pad_x)
            crop_y2 = min(frame.shape[0], y2 + pad_y)
            
            person_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if person_crop.size == 0:
                refined_persons.append(person)
                continue
            
            crop_h, crop_w = person_crop.shape[:2]
            
            try:
                rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                pose_results = self.pose.process(rgb)
                
                if pose_results.pose_landmarks:
                    # Collect relevant keypoints for bounding box refinement
                    # Key points: nose (0), shoulders (11-12), elbows (13-14), wrists (15-16),
                    #            hips (23-24), knees (25-26), ankles (27-28)
                    relevant_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
                    
                    keypoints_x = []
                    keypoints_y = []
                    
                    for idx in relevant_indices:
                        lm = pose_results.pose_landmarks.landmark[idx]
                        
                        # Only use high-visibility keypoints
                        if lm.visibility > 0.5:
                            # Transform coordinates back to full frame
                            px = int(lm.x * crop_w) + crop_x1
                            py = int(lm.y * crop_h) + crop_y1
                            keypoints_x.append(px)
                            keypoints_y.append(py)
                    
                    if len(keypoints_x) >= 4:  # Need at least 4 keypoints
                        # Calculate new tight bounding box from keypoints
                        new_x1 = min(keypoints_x)
                        new_y1 = min(keypoints_y)
                        new_x2 = max(keypoints_x)
                        new_y2 = max(keypoints_y)
                        
                        # Apply small margin (5-10 pixels) for coverage
                        margin = 8
                        new_x1 = max(0, new_x1 - margin)
                        new_y1 = max(0, new_y1 - margin)
                        new_x2 = min(frame.shape[1], new_x2 + margin)
                        new_y2 = min(frame.shape[0], new_y2 + margin)
                        
                        # Calculate new center and foot center
                        new_cx = (new_x1 + new_x2) // 2
                        new_cy = (new_y1 + new_y2) // 2
                        new_foot_cx = (new_x1 + new_x2) // 2
                        new_foot_cy = new_y2
                        
                        # Create refined detection
                        refined_person = PersonDetection(
                            center=(new_cx, new_cy),
                            foot_center=(new_foot_cx, new_foot_cy),
                            bbox=(new_x1, new_y1, new_x2, new_y2),
                            confidence=person.confidence,
                            track_id=person.track_id
                        )
                        
                        # Copy skeleton landmarks for display if skeleton drawing is enabled
                        if person.skeleton_landmarks:
                            refined_person.skeleton_landmarks = person.skeleton_landmarks
                        
                        refined_persons.append(refined_person)
                    else:
                        # Not enough keypoints, keep original bbox
                        refined_persons.append(person)
                else:
                    # No skeleton detected, keep original bbox
                    refined_persons.append(person)
            except Exception as e:
                print(f"[Skeleton Refine] Error: {e}")
                refined_persons.append(person)
        
        return refined_persons
    
    def detect(self, frame: np.ndarray, draw_skeleton: bool = False) -> Tuple[List[PersonDetection], np.ndarray]:
        if frame is None or not self._loaded:
            return [], frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        
        if download_manager.get_status()[0]:
            return [], frame.copy()
        
        persons = []
        output = frame.copy()
        h, w = frame.shape[:2]
        
        print(f"[DEBUG] Frame size: {w}x{h}")
        
        # Calculate adaptive drawing parameters based on resolution
        # High resolution (2K/4K) needs thicker lines and larger fonts
        max_dim = max(w, h)
        if max_dim >= 2000:  # 2K/4K resolution
            bbox_thickness = 6
            font_scale = 1.2
            font_thickness = 3
            circle_radius = 12
            print(f"[DEBUG] High resolution detected: using thick lines (thickness={bbox_thickness}, font={font_scale})")
        elif max_dim >= 1280:  # HD resolution
            bbox_thickness = 3
            font_scale = 0.7
            font_thickness = 2
            circle_radius = 8
            print(f"[DEBUG] HD resolution detected: using medium lines (thickness={bbox_thickness}, font={font_scale})")
        else:  # SD/Low resolution
            bbox_thickness = 2
            font_scale = 0.5
            font_thickness = 1
            circle_radius = 5
            print(f"[DEBUG] SD resolution detected: using normal lines (thickness={bbox_thickness}, font={font_scale})")
        
        # Check if frame is vertical stack (h > w indicates V380 dual-lens vertical stacking)
        # This works for ANY vertical stack resolution (1280x720, 2304x2592, etc.)
        if h > w:
            print(f"[DEBUG] Entering SPLIT FRAME mode (vertical stack detected: {w}x{h})")
            try:
                persons = self.detect_split_frame(frame)
                
                # Draw detections for split frame
                for p in persons:
                    x1, y1, x2, y2 = p.bbox
                    
                    # Draw bounding box with adaptive thickness
                    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), bbox_thickness)
                    
                    # Draw confidence label with adaptive font
                    label = f"Person {p.confidence:.0%}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    cv2.rectangle(output, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
                    cv2.putText(output, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
                    
                    # Draw foot marker with adaptive radius
                    cv2.circle(output, p.foot_center, circle_radius, (255, 0, 255), -1)
                    
                    # Draw professional skeleton if available
                    if draw_skeleton and p.skeleton_landmarks:
                        self._draw_professional_skeleton(output, p.skeleton_landmarks, p.bbox)
                
                # Run skeleton detection for each person if enabled
                if draw_skeleton and MEDIAPIPE_AVAILABLE and persons:
                    self._init_pose()
                    if self.pose:
                        for person in persons:
                            skeleton = self._detect_skeleton_for_person(frame, person.bbox)
                            if skeleton:
                                person.skeleton_landmarks = skeleton
                
                return persons, output
            except Exception as e:
                print(f"[Detector] Split frame error: {e}")
                persons = []
        
        # Run YOLO detection (non-split frame)
        print(f"[DEBUG] Running SINGLE camera detection (non-split frame)")
        try:
            with self._lock:
                results = self.model(frame, conf=self.confidence, classes=[0], verbose=False)
            
            for result in results:
                if result.boxes is None:
                    continue
                for i, box in enumerate(result.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    person = PersonDetection(
                        center=(cx, cy),
                        foot_center=(cx, y2),
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        track_id=i
                    )
                    persons.append(person)
        except MemoryError:
            print("[Detector] Memory error")
            gc.collect()
        except Exception as e:
            print(f"[Detector] Detection error: {e}")
        
        # Run skeleton detection for each person if enabled
        if draw_skeleton and MEDIAPIPE_AVAILABLE and persons:
            self._init_pose()
            if self.pose:
                for person in persons:
                    skeleton = self._detect_skeleton_for_person(frame, person.bbox)
                    if skeleton:
                        person.skeleton_landmarks = skeleton
        
        # Draw detections
        for p in persons:
            x1, y1, x2, y2 = p.bbox
            
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence label
            label = f"Person {p.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(output, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Draw foot marker
            cv2.circle(output, p.foot_center, 5, (255, 0, 255), -1)
            
            # Draw professional skeleton if available
            if draw_skeleton and p.skeleton_landmarks:
                self._draw_professional_skeleton(output, p.skeleton_landmarks, p.bbox)
        
        return persons, output
    
    def _detect_skeleton_for_person(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> List[SkeletonLandmark]:
        """Detect skeleton for a specific person's bounding box."""
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        
        # Expand bbox slightly for better pose detection
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        
        # Crop person region
        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return []
        
        crop_h, crop_w = person_crop.shape[:2]
        
        try:
            rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            pose_results = self.pose.process(rgb)
            
            if pose_results.pose_landmarks:
                landmarks = []
                for idx, lm in enumerate(pose_results.pose_landmarks.landmark):
                    # Transform coordinates back to original frame
                    px = int(lm.x * crop_w) + x1
                    py = int(lm.y * crop_h) + y1
                    landmarks.append(SkeletonLandmark(
                        x=px, y=py,
                        visibility=lm.visibility,
                        name=f"point_{idx}"
                    ))
                return landmarks
        except Exception as e:
            print(f"[Skeleton] Error: {e}")
        
        return []
    
    def _draw_professional_skeleton(self, frame: np.ndarray, landmarks: List[SkeletonLandmark], bbox: Tuple[int, int, int, int]):
        """Draw a professional-looking skeleton."""
        if len(landmarks) < 33:
            return
        
        x1, y1, x2, y2 = bbox
        
        # Draw connections with gradient colors
        for start_idx, end_idx, color in self.SKELETON_CONNECTIONS:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                
                # Only draw if both points are visible and within bbox
                if start.visibility > 0.5 and end.visibility > 0.5:
                    # Check if points are roughly within the person's area
                    if self._point_near_bbox(start.x, start.y, bbox) and self._point_near_bbox(end.x, end.y, bbox):
                        # Draw glow effect
                        cv2.line(frame, (start.x, start.y), (end.x, end.y), 
                                (color[0]//3, color[1]//3, color[2]//3), 6)
                        # Draw main line
                        cv2.line(frame, (start.x, start.y), (end.x, end.y), color, 3)
                        # Draw bright center
                        cv2.line(frame, (start.x, start.y), (end.x, end.y), 
                                (min(255, color[0]+50), min(255, color[1]+50), min(255, color[2]+50)), 1)
        
        # Draw joints with glow
        for idx, lm in enumerate(landmarks):
            if lm.visibility > 0.5 and self._point_near_bbox(lm.x, lm.y, bbox):
                # Determine joint color based on body part
                if idx in [0, 2, 5, 7, 8]:  # Head
                    color = self.JOINT_COLORS['head']
                    radius = 4
                elif idx in [11, 12, 23, 24]:  # Torso
                    color = self.JOINT_COLORS['torso']
                    radius = 5
                elif idx in [13, 14, 15, 16]:  # Arms
                    color = self.JOINT_COLORS['arm']
                    radius = 4
                elif idx in [25, 26, 27, 28]:  # Legs
                    color = self.JOINT_COLORS['leg']
                    radius = 5
                else:
                    continue
                
                # Draw glow
                cv2.circle(frame, (lm.x, lm.y), radius + 3, (color[0]//3, color[1]//3, color[2]//3), -1)
                # Draw joint
                cv2.circle(frame, (lm.x, lm.y), radius, color, -1)
                # Draw highlight
                cv2.circle(frame, (lm.x, lm.y), radius - 1, (255, 255, 255), 1)
    
    def _point_near_bbox(self, x: int, y: int, bbox: Tuple[int, int, int, int], margin: float = 0.3) -> bool:
        """Check if point is near the bounding box."""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        
        # Allow some margin outside bbox
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        
        return (x1 - margin_x <= x <= x2 + margin_x and 
                y1 - margin_y <= y <= y2 + margin_y)


class FaceRecognitionEngine:
    """Face recognition."""
    
    def __init__(self, config: Config):
        self.config = config
        self.known_faces = {}
        self.known_names = []
        self._lock = threading.Lock()
        self._load_faces()
    
    def _load_faces(self):
        if not FACE_RECOGNITION_AVAILABLE:
            return
        
        with self._lock:
            self.known_faces.clear()
            self.known_names.clear()
            
            if self.config.TRUSTED_FACES_DIR.exists():
                for fp in self.config.TRUSTED_FACES_DIR.iterdir():
                    if fp.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                        try:
                            img = cv2.imread(str(fp))
                            if img is None:
                                continue
                            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            locs = face_recognition.face_locations(rgb, model="hog")
                            if locs:
                                dest = self.config.FIXED_IMAGES_DIR / fp.name
                                if not dest.exists():
                                    shutil.copy2(str(fp), str(dest))
                        except:
                            pass
            
            if self.config.FIXED_IMAGES_DIR.exists():
                for fp in self.config.FIXED_IMAGES_DIR.iterdir():
                    if fp.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                        try:
                            img = cv2.imread(str(fp))
                            if img is None:
                                continue
                            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            locs = face_recognition.face_locations(rgb, model="hog")
                            if locs:
                                enc = face_recognition.face_encodings(rgb, locs)
                                if enc:
                                    self.known_faces[fp.stem] = enc[0]
                                    self.known_names.append(fp.stem)
                        except:
                            pass
            
            print(f"[Faces] Loaded {len(self.known_names)}")
    
    def reload_faces(self):
        self._load_faces()
    
    def recognize_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        if not FACE_RECOGNITION_AVAILABLE or frame is None:
            return []
        
        results = []
        try:
            h, w = frame.shape[:2]
            scale = 0.2  # Reduced scale for faster processing
            small = cv2.resize(frame, (int(w*scale), int(h*scale)))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            
            locs = face_recognition.face_locations(rgb, model="hog")
            if not locs:
                return []
            
            encs = face_recognition.face_encodings(rgb, locs)
            
            with self._lock:
                for (top, right, bottom, left), enc in zip(locs, encs):
                    top = int(top / scale)
                    right = int(right / scale)
                    bottom = int(bottom / scale)
                    left = int(left / scale)
                    
                    name = "Unknown"
                    conf = 0.0
                    trusted = False
                    
                    if self.known_names:
                        known_encs = list(self.known_faces.values())
                        matches = face_recognition.compare_faces(known_encs, enc, self.config.FACE_MATCH_TOLERANCE)
                        dists = face_recognition.face_distance(known_encs, enc)
                        
                        if len(dists) > 0:
                            best = np.argmin(dists)
                            if matches[best]:
                                name = self.known_names[best]
                                conf = 1.0 - dists[best]
                                trusted = True
                    
                    results.append(FaceDetection(
                        name=name, confidence=conf, is_trusted=trusted,
                        bbox=(left, top, right, bottom)
                    ))
        except Exception as e:
            print(f"[Faces] Error: {e}")
        
        return results


class MotionDetector:
    """Motion detection with heat map and zone awareness - Enhanced for better accuracy."""
    
    def __init__(self, config: Config):
        self.config = config
        self.prev_frame = None
        self.heat_map = None
        self.frame_size = None
        self.threshold = config.MOTION_THRESHOLD
        self.min_area = config.MOTION_MIN_AREA
        self.motion_history = deque(maxlen=5)  # Track recent motion frames
        self.motion_active = False
        self.last_motion_time = 0
    
    def set_sensitivity(self, sensitivity: Sensitivity):
        settings = Config.get_sensitivity_settings(sensitivity)
        self.threshold = settings.get('motion_threshold', 20)
        self.min_area = settings.get('motion_min_area', 300)
    
    def detect(self, frame: np.ndarray) -> Tuple[bool, List[Tuple[int, int, int, int]]]:
        if frame is None:
            return False, []
        
        h, w = frame.shape[:2]
        size = (w, h)
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Initialize or reset if size changed
        if self.frame_size != size or self.prev_frame is None:
            self.prev_frame = gray
            self.heat_map = np.zeros((h, w), dtype=np.float32)
            self.frame_size = size
            self.motion_history.clear()
            return False, []
        
        # Calculate absolute difference
        delta = cv2.absdiff(self.prev_frame, gray)
        
        # Apply threshold - more sensitive lower threshold
        thresh = cv2.threshold(delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Dilate to fill in holes and make motion regions more continuous
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and create motion regions
        regions = []
        has_motion = False
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.min_area:
                has_motion = True
                x, y, cw, ch = cv2.boundingRect(cnt)
                
                # Filter out very narrow regions (likely noise)
                aspect_ratio = cw / ch if ch > 0 else 0
                if aspect_ratio > 0.1 and aspect_ratio < 10:  # Reasonable aspect ratio
                    regions.append((x, y, x + cw, y + ch))
        
        # Update heat map with decay
        if has_motion:
            self.heat_map = self.heat_map * 0.85 + thresh.astype(np.float32) * 0.15
            self.last_motion_time = time.time()
            self.motion_history.append(True)
        else:
            self.heat_map = self.heat_map * 0.95  # Faster decay when no motion
            self.motion_history.append(False)
        
        # Update previous frame
        self.prev_frame = gray
        
        # Determine if motion is active (based on recent history)
        recent_motion_count = sum(1 for m in self.motion_history if m)
        self.motion_active = recent_motion_count >= 2  # Motion if 2+ frames in last 5
        
        return self.motion_active, regions
    
    def get_heat_map(self) -> Optional[np.ndarray]:
        """Get motion heat map."""
        if self.heat_map is None:
            return None
        return np.clip(self.heat_map, 0, 255).astype(np.uint8)
    
    def reset(self):
        """Reset motion detector state."""
        self.prev_frame = None
        self.heat_map = None
        self.frame_size = None


class DetectionThread(threading.Thread):
    """Detection thread with skeleton and motion support."""
    
    def __init__(self, person_detector: PersonDetector, motion_detector: MotionDetector):
        super().__init__(daemon=True)
        self.person_detector = person_detector
        self.motion_detector = motion_detector
        self._input_queue = Queue(maxsize=2)
        self._running = False
        self._result_lock = threading.Lock()
        
        self.last_persons = []
        self.last_motion = False
        self.last_motion_regions = []
        self.last_frame = None
        
        self.draw_skeleton = False
    
    def run(self):
        self._running = True
        while self._running:
            try:
                frame = self._input_queue.get(timeout=0.1)
                
                if frame is None or frame.size == 0:
                    continue
                
                if len(frame.shape) != 3 or frame.shape[2] != 3:
                    continue
                
                if download_manager.get_status()[0]:
                    with self._result_lock:
                        self.last_frame = frame.copy()
                    continue
                
                try:
                    # Detect persons with optional skeleton
                    persons, processed = self.person_detector.detect(frame, self.draw_skeleton)
                    
                    # Detect motion
                    motion, regions = self.motion_detector.detect(frame)
                    
                    with self._result_lock:
                        self.last_persons = persons
                        self.last_motion = motion
                        self.last_motion_regions = regions
                        self.last_frame = processed
                except MemoryError:
                    print("[Detection] Memory error")
                    gc.collect()
                except Exception as e:
                    print(f"[Detection] Error: {e}")
                    
            except Empty:
                continue
            except Exception:
                pass
    
    def stop(self):
        self._running = False
    
    def submit(self, frame: np.ndarray):
        if frame is None:
            return
        try:
            while not self._input_queue.empty():
                try:
                    self._input_queue.get_nowait()
                except:
                    break
            self._input_queue.put_nowait(frame)
        except:
            pass
    
    def get_results(self) -> dict:
        with self._result_lock:
            return {
                'persons': list(self.last_persons) if self.last_persons else [],
                'motion': self.last_motion,
                'motion_regions': list(self.last_motion_regions) if self.last_motion_regions else [],
                'frame': self.last_frame.copy() if self.last_frame is not None else None
            }
