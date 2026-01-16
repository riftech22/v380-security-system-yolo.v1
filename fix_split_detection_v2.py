#!/usr/bin/env python3
"""Fixed and robust split frame detection implementation."""

import re

print("[1/5] Reading detectors.py...", flush=True)
with open('detectors.py', 'r') as f:
    content = f.read()

# Create backup
print("[2/5] Creating backup...", flush=True)
with open('detectors.py.backup4', 'w') as f:
    f.write(content)

# Check if detect_split_frame already exists
if 'def detect_split_frame' in content:
    print("⚠️  detect_split_frame() already exists!", flush=True)
    print("🔍 Checking if it's being used...", flush=True)
    
    if 'self.detect_split_frame(frame)' in content:
        print("✅ detect_split_frame() is being called", flush=True)
    else:
        print("❌ detect_split_frame() exists but NOT being called!", flush=True)
        print("🔧 Will fix detect() method to use it", flush=True)
else:
    print("❌ detect_split_frame() NOT found!", flush=True)

print("\n[3/5] Finding detect() method...", flush=True)

# Find the detect method signature
detect_method_pattern = r'(    def detect\(self, frame: np\.ndarray, draw_skeleton: bool = False\) -> Tuple\[List\[PersonDetection\], np\.ndarray\]:)'
match = re.search(detect_method_pattern, content)

if not match:
    print("❌ Could not find detect() method signature!", flush=True)
    exit(1)

detect_start = match.start()
print(f"✅ Found detect() method at position {detect_start}", flush=True)

# Find the start of the detect method body
after_signature = content.find(':', detect_start)
if after_signature == -1:
    print("❌ Could not find method body!", flush=True)
    exit(1)

# Find where the detection logic starts (after the docstring and initial checks)
# Look for the first actual detection code
person_search = content.find('persons = []', after_signature)
if person_search == -1:
    person_search = content.find('        persons = []', after_signature)

if person_search == -1:
    print("❌ Could not find 'persons = []' in detect() method!", flush=True)
    exit(1)

print(f"✅ Found detection logic at position {person_search}", flush=True)

# Check if detect_split_frame method already exists
split_frame_method = '''
    def detect_split_frame(self, frame, top_conf=0.25, bottom_conf=0.15):
        """Detect persons in split frame (2 cameras).
        
        Args:
            frame: Input frame (1280x720 with vertical split)
            top_conf: Confidence threshold for top camera (default: 0.25)
            bottom_conf: Confidence threshold for bottom camera (default: 0.15)
            
        Returns:
            List of PersonDetection objects with adjusted coordinates
        """
        h, w = frame.shape[:2]
        mid_y = h // 2
        
        # Split frame
        top_frame = frame[:mid_y, :]
        bottom_frame = frame[mid_y:, :]
        
        all_persons = []
        track_id = 0
        
        # Detect in top camera
        if top_frame.size > 0:
            try:
                with self._lock:
                    results_top = self.model(top_frame, conf=top_conf, classes=[0], verbose=False)
                
                if results_top and results_top[0].boxes:
                    for box in results_top[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf = float(box.conf[0])
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        
                        person = PersonDetection(
                            center=(cx, cy),
                            foot_center=(cx, y2),
                            bbox=(x1, y1, x2, y2),
                            confidence=conf,
                            track_id=track_id
                        )
                        all_persons.append(person)
                        track_id += 1
            except Exception as e:
                print(f"[Split-Top] Error: {e}")
        
        # Detect in bottom camera
        if bottom_frame.size > 0:
            try:
                with self._lock:
                    results_bottom = self.model(bottom_frame, conf=bottom_conf, classes=[0], verbose=False)
                
                if results_bottom and results_bottom[0].boxes:
                    for box in results_bottom[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf = float(box.conf[0])
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        
                        # Adjust coordinates to full frame (bottom region: mid_y-h)
                        person = PersonDetection(
                            center=(cx, cy + mid_y),
                            foot_center=(cx, y2 + mid_y),
                            bbox=(x1, y1 + mid_y, x2, y2 + mid_y),
                            confidence=conf,
                            track_id=track_id
                        )
                        all_persons.append(person)
                        track_id += 1
            except Exception as e:
                print(f"[Split-Bottom] Error: {e}")
        
        return all_persons
'''

# Insert detect_split_frame method before detect method
if 'def detect_split_frame' not in content:
    print("\n[4/5] Adding detect_split_frame() method...", flush=True)
    content = content[:detect_start] + split_frame_method + '\n\n' + content[detect_start:]
    print("✅ detect_split_frame() method added!", flush=True)
else:
    print("\n[4/5] detect_split_frame() already exists, skipping...", flush=True)

# Now update detect() method to use split frame
print("[5/5] Updating detect() method to use split frame...", flush=True)

# Find the persons = [] line again (might have moved)
person_search_new = content.find('persons = []', after_signature)
if person_search_new == -1:
    person_search_new = content.find('        persons = []', after_signature)

if person_search_new == -1:
    print("❌ Could not find 'persons = []' after adding method!", flush=True)
    exit(1)

# Insert split frame check before persons = []
split_check = '''        # Check if frame is split (1280x720 resolution)
        h, w = frame.shape[:2]
        
        # Use split frame detection for 1280x720 resolution
        if w == 1280 and h == 720:
            try:
                persons = self.detect_split_frame(frame)
                output = frame.copy()
                
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
                    if self.draw_skeleton and p.skeleton_landmarks:
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
                # Fall through to regular detection
        else:
        
'''

content = content[:person_search_new] + split_check + content[person_search_new:]

# Write updated content
print("\n✅ Writing updated detectors.py...", flush=True)
with open('detectors.py', 'w') as f:
    f.write(content)

print("\n" + "=" * 70, flush=True)
print("✅ Split Frame Detection Implementation Complete!", flush=True)
print("=" * 70, flush=True)
print()
print("📋 Changes made:", flush=True)
print("   1. Added/verified detect_split_frame() method", flush=True)
print("   2. Updated detect() to auto-detect split frame (1280x720)", flush=True)
print("   3. Top camera: confidence 0.25", flush=True)
print("   4. Bottom camera: confidence 0.15 (more sensitive)", flush=True)
print()
print("📋 Next steps:", flush=True)
print("   1. Check syntax: python -m py_compile detectors.py", flush=True)
print("   2. sudo systemctl restart security-system-web", flush=True)
print("   3. sudo systemctl status security-system-web", flush=True)
print("   4. Test at: http://10.26.27.104:8080/web.html", flush=True)
print()
print("💾 Backup created: detectors.py.backup4", flush=True)
print()
