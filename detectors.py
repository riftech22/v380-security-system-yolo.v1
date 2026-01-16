#!/usr/bin/env python3
"""Detection modules - Professional skeleton, accurate zone detection."""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import threading
from queue import Queue, Empty
import shutil
import gc
import logging

# Setup logging
logger = logging.getLogger(__name__)

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

# Try to import MediaPipe for skeleton detection
MEDIAPIPE_AVAILABLE = False
_mp_solutions = None

try:
    import mediapipe as mp
    
    # MediaPipe 0.10.x+ uses mp.solutions.pose.Pose (lowercase 'pose' module)
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
        _mp_solutions = mp.solutions.pose
        MEDIAPIPE_AVAILABLE = True
        logger.info("[MediaPipe] Available (0.10.x+ API) - skeleton detection enabled")
    elif hasattr(mp, 'solutions'):
        # Fallback to old MediaPipe API (0.10.0 and earlier)
        _mp_solutions = mp.solutions
        MEDIAPIPE_AVAILABLE = True
        logger.info("[MediaPipe] Available (old API) - skeleton detection enabled")
    else:
        logger.warning("[MediaPipe] mp.solutions not available, skeleton detection disabled")
except ImportError:
    logger.warning("[MediaPipe] Not installed, skeleton detection disabled")
except Exception as e:
    logger.error(f"[MediaPipe] Error: {e}")
    MEDIAPIPE_AVAILABLE = False


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
        
        # MediaPipe pose for skeleton (use global reference)
        self.pose = None
        self.mp_pose = _mp_solutions  # Use global reference
        self._mediapipe_available = MEDIAPIPE_AVAILABLE
        self.draw_skeleton = False  # Skeleton drawing toggle
        
        if YOLO_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        try:
            download_manager.start_download(self.model_name)
            # Import logger here to avoid circular import
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"[Detector] Loading {self.model_name}...")
            self.model = YOLO(self.model_name)
            self._loaded = True
            logger.info("[Detector] Loaded")
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"[Detector] Error: {e}")
        finally:
            download_manager.end_download()
    
    def _init_pose(self):
        """Lazy init MediaPipe pose."""
        if self.pose is None and self._mediapipe_available and self.mp_pose:
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
    
    def _preprocess_frame(self, frame: np.ndarray, target_size: tuple = (1280, 720)) -> np.ndarray:
        """Preprocess frame for better detection.
        
        Args:
            frame: Input frame (can be any resolution)
            target_size: Target resolution (default: 1280x720)
            
        Returns:
            Preprocessed frame at target_size
        """
        # Step 1: Resize to target resolution
        h, w = frame.shape[:2]
        if (w, h) != target_size:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Step 2: Calculate brightness statistics
        brightness = np.mean(frame)
        std_dev = np.std(frame)
        
        # Step 3: Brightness adjustment
        if brightness > 120:  # Overexposed
            # Darken frame
            frame = cv2.convertScaleAbs(frame, alpha=0.7, beta=-30)
        elif brightness < 50:  # Underexposed
            # Brighten frame
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
        """Detect persons in split frame (2 cameras) with Frigate-style techniques.
        
        Args:
            frame: Input frame (can be any resolution, will be resized to 1280x720)
            top_conf: Confidence threshold for top camera (Frigate: 0.4)
            bottom_conf: Confidence threshold for bottom camera (Frigate: 0.4)
            
        Returns:
            List of PersonDetection objects with adjusted coordinates
        """
        # Preprocess entire frame (resize + brightness + CLAHE)
        frame_processed = self._preprocess_frame(frame)
        
        h, w = frame_processed.shape[:2]
        mid_y = h // 2
        
        # Split processed frame
        top_frame = frame_processed[:mid_y, :]
        bottom_frame = frame_processed[mid_y:, :]
        
        top_persons = []
        bottom_persons = []
        
        # Frigate-style filters (from objects.py FilterConfig):
        min_area = 1000
        max_area = 200000
        min_ratio = 0.3  # Person aspect ratio (width/height)
        max_ratio = 1.0
        
        # Detect in top camera (already enhanced by preprocessing)
        if top_frame.size > 0:
            try:
                with self._lock:
                    results_top = self.model(top_frame, conf=top_conf, classes=[0], verbose=False)
                
                if results_top and results_top[0].boxes:
                    for box in results_top[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf = float(box.conf[0])
                        
                        # Calculate area
                        area = (x2 - x1) * (y2 - y1)
                        
                        # Frigate-style area filtering
                        if area < min_area or area > max_area:
                            continue
                        
                        # Calculate aspect ratio (width/height)
                        width = x2 - x1
                        height = y2 - y1
                        aspect_ratio = width / height if height > 0 else 0
                        
                        # Frigate-style aspect ratio filtering (person: 0.3-1.0)
                        if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
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
        if bottom_frame.size > 0:
            try:
                with self._lock:
                    results_bottom = self.model(bottom_frame, conf=bottom_conf, classes=[0], verbose=False)
                
                if results_bottom and results_bottom[0].boxes:
                    for box in results_bottom[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf = float(box.conf[0])
                        
                        # Calculate area
                        area = (x2 - x1) * (y2 - y1)
                        
                        # Frigate-style area filtering
                        if area < min_area or area > max_area:
                            continue
                        
                        # Calculate aspect ratio (width/height)
                        width = x2 - x1
                        height = y2 - y1
                        aspect_ratio = width / height if height > 0 else 0
                        
                        # Frigate-style aspect ratio filtering (person: 0.3-1.0)
                        if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
                            continue
                        
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        
                        # Adjust coordinates to full frame (bottom region: mid_y-h)
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
        
        return all_persons
    
    def detect(self, frame: np.ndarray, draw_skeleton: bool = False) -> Tuple[List[PersonDetection], np.ndarray]:
        if frame is None or not self._loaded:
            return [], frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        
        if download_manager.get_status()[0]:
            return [], frame.copy()
        
        persons = []
        output = frame.copy()
        h, w = frame.shape[:2]
        
        # Check if frame is split (1280x720 resolution)
        if w == 1280 and h == 720:
            try:
                persons = self.detect_split_frame(frame)
                
                # Draw detections for split frame
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
                
                # Run skeleton detection for each person if enabled
                if self.draw_skeleton and MEDIAPIPE_AVAILABLE and persons:
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
        
        # Run YOLO detection
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
        # Use self.draw_skeleton instead of parameter for toggle support
        if self.draw_skeleton and MEDIAPIPE_AVAILABLE and persons:
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
                    # Check if points are roughly within person's area
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
        """Check if point is near bounding box."""
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
    """Motion detection with heat map and zone awareness."""
    
    def __init__(self, config: Config):
        self.config = config
        self.prev_frame = None
        self.heat_map = None
        self.frame_size = None
        self.threshold = config.MOTION_THRESHOLD
        self.min_area = config.MOTION_MIN_AREA
    
    def set_sensitivity(self, sensitivity: Sensitivity):
        settings = Config.get_sensitivity_settings(sensitivity)
        self.threshold = settings.get('motion_threshold', 20)
        self.min_area = settings.get('motion_min_area', 300)
    
    def detect(self, frame: np.ndarray) -> Tuple[bool, List[Tuple[int, int, int, int]]]:
        if frame is None:
            return False, []
        
        h, w = frame.shape[:2]
        size = (w, h)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Initialize or reset if size changed
        if self.frame_size != size or self.prev_frame is None:
            self.prev_frame = gray
            self.heat_map = np.zeros((h, w), dtype=np.float32)
            self.frame_size = size
            return False, []
        
        delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Update heat map with decay
        self.heat_map = self.heat_map * 0.9 + thresh.astype(np.float32) * 0.1
        
        self.prev_frame = gray
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        has_motion = False
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > self.min_area:
                has_motion = True
                x, y, cw, ch = cv2.boundingRect(cnt)
                regions.append((x, y, x + cw, y + ch))
        
        return has_motion, regions
    
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
