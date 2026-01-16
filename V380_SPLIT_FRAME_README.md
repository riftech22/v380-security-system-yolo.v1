# V380 Dual-Lens Split Frame Detection - Implementation Guide

## Overview

This document explains the enhanced split frame detection implementation for V380 dual-lens cameras. The system has been optimized to handle the unique vertical stacking format where two camera views are combined into a single video frame.

## V380 Camera Anatomy

### Frame Structure
```
┌────────────────────────────────────┐
│       TOP CAMERA (Wide Angle)    │ 1280x360 px
│       Coordinates: (0,0) to      │
│       (1280,360)                │
│                                    │
│                                    │
├────────────────────────────────────┤
│       BOTTOM CAMERA (PTZ)         │ 1280x360 px
│       Coordinates: (0,360) to    │
│       (1280,720)                 │
│                                    │
│                                    │
└────────────────────────────────────┘
Total Frame: 1280 x 720 px (HD)
```

### Key Characteristics

1. **Total Resolution**: 1280 x 720 pixels (HD)
2. **Split Point**: Horizontal line at y=360 (middle of frame)
3. **Top Camera**: Fixed wide-angle lens for area monitoring
4. **Bottom Camera**: PTZ (Pan-Tilt-Zoom) lens for tracking details

### Aspect Ratio Challenge

Each split has an extreme aspect ratio:
- **Split dimensions**: 1280x360 pixels
- **Aspect ratio**: 1280:360 = **3.5:1** (very wide!)
- **Problem**: Without proper handling, people appear distorted (short and wide)
- **Solution**: Letterboxing maintains aspect ratio during YOLO processing

## Implementation Details

### 1. Frame Normalization

```python
# Step 1: Normalize to standard V380 resolution (1280x720)
# This ensures consistent split point regardless of input resolution
if frame.shape[1] != 1280 or frame.shape[0] != 720:
    frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
```

**Purpose**: Ensures all frames are processed at the same resolution, regardless of camera output variations.

### 2. Frame Splitting

```python
# Step 2: Crop at exact split point (360px for 720p)
split_point = h // 2  # 360px for 720p
top_frame_raw = frame[:split_point, :]  # Wide angle: 1280x360
bottom_frame_raw = frame[split_point:, :]  # PTZ: 1280x360
```

**Purpose**: Separates the dual-camera views into two independent frames for individual processing.

### 3. Individual Preprocessing

Each split is preprocessed separately:
- **Brightness adjustment**: Corrects overexposed/underexposed areas
- **CLAHE enhancement**: Improves contrast for better detection
- **Noise reduction**: Applies Gaussian blur for cleaner detection

```python
top_frame = self._preprocess_frame(top_frame_raw)
bottom_frame = self._preprocess_frame(bottom_frame_raw)
```

### 4. Letterboxing (Critical!)

This is the **most important** step for preventing AI confusion:

```python
def _letterbox_resize(self, frame: np.ndarray, target_size: int = 640) -> np.ndarray:
    """Resize frame with letterboxing (maintain aspect ratio)."""
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
```

**Why This Matters**:
- Without letterboxing: People appear 3.5x wider than normal
- With letterboxing: People maintain normal proportions
- YOLO model was trained on normal human proportions
- Distorted shapes cause AI to fail or produce false positives

### 5. YOLO Detection

Each split is detected independently:

```python
top_frame_detect = self._letterbox_resize(top_frame, DETECTION_SIZE)  # 640x640
bottom_frame_detect = self._letterbox_resize(bottom_frame, DETECTION_SIZE)  # 640x640

results_top = self.model(top_frame_detect, conf=top_conf, classes=[0], verbose=False)
results_bottom = self.model(bottom_frame_detect, conf=bottom_conf, classes=[0], verbose=False)
```

### 6. Coordinate Mapping

After detection, coordinates must be mapped back to the original frame:

```python
# Calculate letterbox offsets
scale = min(DETECTION_SIZE / top_w, DETECTION_SIZE / top_h)
x_offset = (DETECTION_SIZE - new_w) // 2
y_offset = (DETECTION_SIZE - new_h) // 2

# Remove letterbox offset and scale back to original size
x1 = int((x1 - x_offset) / scale)
y1 = int((y1 - y_offset) / scale)
x2 = int((x2 - x_offset) / scale)
y2 = int((y2 - y_offset) / scale)
```

### 7. Frigate-Style Filtering

Advanced filtering reduces false positives:

```python
# Area filtering (pixels)
min_area = 5000    # Reject tiny detections
max_area = 50000   # Reject large bboxes (background)

# Aspect ratio filtering (width/height)
min_ratio = 0.4     # Person minimum ratio
max_ratio = 1.3     # Person maximum ratio

# Frame percentage filtering
if area > frame_area * 0.15:  # Reject if > 15% of frame
    continue
```

### 8. Non-Maximum Suppression (NMS)

Removes duplicate detections:

```python
top_persons = self._apply_nms(top_persons, iou_threshold=0.45, conf_threshold=0.5)
bottom_persons = self._apply_nms(bottom_persons, iou_threshold=0.45, conf_threshold=0.5)
```

- **IoU threshold**: 0.45 (Frigate standard)
- **Confidence threshold**: 0.5 (Frigate standard)

### 9. Skeleton Refinement (Optional)

If MediaPipe is available, skeleton keypoints refine bounding boxes:

```python
if self._mediapipe_available and MEDIAPIPE_AVAILABLE:
    top_persons = self._refine_bboxes_with_skeleton(top_frame, top_persons, offset_y=0)
    bottom_persons = self._refine_bboxes_with_skeleton(bottom_frame, bottom_persons, offset_y=mid_y)
```

**Benefits**:
- Tighter bounding boxes
- Better object separation
- Improved tracking accuracy

### 10. Coordinate Adjustment

Bottom camera coordinates must be adjusted to full frame:

```python
# Top camera: No adjustment needed (already in correct position)
person_top = PersonDetection(
    center=(cx, cy),
    bbox=(x1, y1, x2, y2),
    ...
)

# Bottom camera: Add mid_y offset
person_bottom = PersonDetection(
    center=(cx, cy + mid_y),  # Offset by split point
    bbox=(x1, y1 + mid_y, x2, y2 + mid_y),  # Offset by split point
    ...
)
```

## Debug Output

The enhanced implementation provides detailed logging:

```
[V380 Split] Input frame: 1280x720
[V380 Split] Processing frame: 1280x720
[V380 Split] Split point at y=360
[V380 Split] Top crop: 1280x360, Bottom crop: 1280x360
[V380 Split] Preprocessed top: 1280x360, bottom: 1280x360
[V380 Split] Letterboxed to: 640x640
[Split-Top] REJECT: area=60000 out of range [5000, 50000]
[Split-Top] REJECT: aspect_ratio=2.50 out of range [0.4, 1.3]
[V380 Split] Final result: 2 persons detected (Top: 1, Bottom: 1)
```

## Configuration

### Camera Settings (config.py)

```python
CAMERA_SOURCE = r"rtsp://admin:password@192.168.1.100:554/live/ch00_0"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
```

### Detection Thresholds

```python
YOLO_CONFIDENCE = 0.25  # Default confidence
# Override in detect_split_frame():
# - Top camera (wide angle): top_conf=0.4
# - Bottom camera (PTZ): bottom_conf=0.4
```

## Testing

Run the test script:

```bash
python3 test_v380_split.py
```

This will:
1. Create a synthetic 1280x720 split frame
2. Test split frame detection
3. Verify letterboxing
4. Check coordinate mapping
5. Save results to `test_v380_frame.jpg` and `test_v380_result.jpg`

## Troubleshooting

### Issue: No detections

**Possible causes**:
1. Confidence threshold too high
2. Frame not properly normalized to 1280x720
3. Lighting conditions poor

**Solutions**:
```python
# Lower confidence threshold
persons = detector.detect_split_frame(frame, top_conf=0.25, bottom_conf=0.25)

# Check frame size
print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")
# Should be: 1280x720
```

### Issue: False positives

**Possible causes**:
1. Aspect ratio filtering too loose
2. Area filtering too permissive
3. NMS threshold too high

**Solutions**:
```python
# Adjust filters in detect_split_frame()
min_area = 8000       # Increase minimum area
max_area = 40000       # Decrease maximum area
min_ratio = 0.5       # Tighten aspect ratio
max_ratio = 1.2       # Tighten aspect ratio
```

### Issue: Distorted detections

**Possible causes**:
1. Letterboxing not applied
2. Coordinate mapping incorrect
3. Split point calculation wrong

**Solutions**:
```python
# Ensure letterboxing is used
top_frame_detect = self._letterbox_resize(top_frame, DETECTION_SIZE)

# Check split point
split_point = h // 2  # Should be 360 for 720p

# Verify coordinate mapping
# Top: (x1, y1) -> (x2, y2) where y2 < 360
# Bottom: (x1, y1+360) -> (x2, y2+360)
```

## Performance Optimization

### 1. Reduce Detection Size

```python
DETECTION_SIZE = 512  # Instead of 640 (faster, less accurate)
```

### 2. Disable Skeleton Refinement

```python
# In detect_split_frame(), comment out:
# top_persons = self._refine_bboxes_with_skeleton(...)
# bottom_persons = self._refine_bboxes_with_skeleton(...)
```

### 3. Use Smaller YOLO Model

```python
# In detectors.py __init__():
self.model_name = 'yolov8n.pt'  # Fastest
# Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
```

## Best Practices

1. **Always normalize to 1280x720** before processing
2. **Apply letterboxing** to maintain aspect ratio
3. **Filter detections** using area and aspect ratio
4. **Use NMS** to remove duplicates
5. **Map coordinates correctly** back to original frame
6. **Test with synthetic frames** to verify logic
7. **Monitor debug output** for detection issues
8. **Adjust thresholds** based on environment

## References

- **V380 Camera Specifications**: Dual-lens IP camera with vertical stacking
- **YOLOv8 Documentation**: https://docs.ultralytics.com/
- **Frigate NVR**: Open-source NVR with advanced object detection
- **MediaPipe Pose**: https://google.github.io/mediapipe/

## Support

For issues or questions:
1. Check debug output logs
2. Run test script: `python3 test_v380_split.py`
3. Review this documentation
4. Check V380 camera settings (resolution, frame rate)

## Changelog

### Version 2.0 (Current)
- Enhanced split frame handling with letterboxing
- Added detailed debugging output
- Improved coordinate mapping
- Frigate-style filtering (area, aspect ratio)
- Skeleton refinement with MediaPipe
- Comprehensive documentation

### Version 1.0 (Previous)
- Basic split frame detection
- Simple cropping
- No letterboxing (caused distortion)
