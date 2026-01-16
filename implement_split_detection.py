#!/usr/bin/env python3
"""
Implementation script untuk split frame detection di production code.

Script ini akan:
1. Menambahkan split frame logic ke detectors.py
2. Menambahkan adaptive confidence threshold
3. Merge bounding boxes ke full frame coordinates
"""

import sys

# Read detectors.py
print("[1/5] Reading detectors.py...", flush=True)
with open('detectors.py', 'r') as f:
    content = f.read()

# Backup original
print("[2/5] Creating backup...", flush=True)
with open('detectors.py.backup', 'w') as f:
    f.write(content)

# Check if split frame already implemented
if 'split_frame' in content.lower():
    print("⚠️  Split frame already implemented in detectors.py!", flush=True)
    sys.exit(0)

# Add split frame imports
print("[3/5] Adding split frame imports...", flush=True)
imports_addition = """
# Split frame detection
import numpy as np
"""

# Find import section and add after cv2 import
import_section = content.find('import cv2')
if import_section != -1:
    insert_pos = content.find('\n', import_section) + 1
    content = content[:insert_pos] + imports_addition + content[insert_pos:]

# Add split frame detection method
print("[4/5] Adding split frame detection method...", flush=True)
split_method = '''
    def detect_split_frame(self, frame, top_conf=0.25, bottom_conf=0.15):
        """
        Detect persons in split frame (2 cameras).
        
        Args:
            frame: Input frame (1280x720 with vertical split)
            top_conf: Confidence threshold for top camera (default: 0.25)
            bottom_conf: Confidence threshold for bottom camera (default: 0.15)
            
        Returns:
            List of bounding boxes with coordinates adjusted to full frame
        """
        h, w = frame.shape[:2]
        mid_y = h // 2
        
        # Split frame
        top_frame = frame[:mid_y, :]
        bottom_frame = frame[mid_y:, :]
        
        all_boxes = []
        
        # Detect in top camera
        if top_frame.size > 0:
            results_top = self.model(top_frame, conf=top_conf, classes=[0], verbose=False)
            for box in results_top[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                
                # Adjust coordinates to full frame (top region: 0-mid_y)
                all_boxes.append({
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2),
                    'conf': conf,
                    'camera': 'top'
                })
        
        # Detect in bottom camera
        if bottom_frame.size > 0:
            results_bottom = self.model(bottom_frame, conf=bottom_conf, classes=[0], verbose=False)
            for box in results_bottom[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                
                # Adjust coordinates to full frame (bottom region: mid_y-h)
                # Offset by mid_y to account for split
                all_boxes.append({
                    'x1': float(x1),
                    'y1': float(y1) + mid_y,
                    'x2': float(x2),
                    'y2': float(y2) + mid_y,
                    'conf': conf,
                    'camera': 'bottom'
                })
        
        return all_boxes
'''

# Find the end of PersonDetector class and add method
class_end = content.find('\nclass ')
if class_end != -1:
    # Find the last line before next class or end of file
    insert_pos = class_end
    content = content[:insert_pos] + split_method + '\n' + content[insert_pos:]

# Update detect() method to use split frame
print("[5/5] Updating detect() method...", flush=True)
old_detect = '''    def detect(self, frame):
        """Detect persons in frame."""
        if self.model is None:
            self._load_model()
        
        if self.model is None:
            return []
        
        results = self.model(frame, conf=self.confidence_threshold, classes=[0], verbose=False)
        detections = []
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            
            # Filter by confidence threshold
            if conf >= self.confidence_threshold:
                detections.append({
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2),
                    'conf': conf
                })
        
        return detections'''

new_detect = '''    def detect(self, frame):
        """Detect persons in frame."""
        if self.model is None:
            self._load_model()
        
        if self.model is None:
            return []
        
        # Check if frame is split (assume 1280x720 resolution)
        h, w = frame.shape[:2]
        
        # Use split frame detection for 1280x720 resolution
        if w == 1280 and h == 720:
            return self.detect_split_frame(frame)
        
        # Use regular detection for other resolutions
        results = self.model(frame, conf=self.confidence_threshold, classes=[0], verbose=False)
        detections = []
        
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            
            # Filter by confidence threshold
            if conf >= self.confidence_threshold:
                detections.append({
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2),
                    'conf': conf
                })
        
        return detections'''

if old_detect in content:
    content = content.replace(old_detect, new_detect)
    print("✅ detect() method updated with split frame logic!", flush=True)
else:
    print("⚠️  Could not find detect() method to update!", flush=True)

# Write updated content
print("✅ Writing updated detectors.py...", flush=True)
with open('detectors.py', 'w') as f:
    f.write(content)

print()
print("=" * 70, flush=True)
print("✅ Split Frame Detection Implementation Complete!", flush=True)
print("=" * 70, flush=True)
print()
print("📋 Changes made:", flush=True)
print("   1. Added split frame detection method", flush=True)
print("   2. Updated detect() to auto-detect split frame", flush=True)
print("   3. Top camera: confidence 0.25", flush=True)
print("   4. Bottom camera: confidence 0.15 (more sensitive)", flush=True)
print()
print("📋 Next steps:", flush=True)
print("   1. Review detectors.py", flush=True)
print("   2. Test the implementation", flush=True)
print("   3. Restart service", flush=True)
print()
print("💾 Backup created: detectors.py.backup", flush=True)
print()
