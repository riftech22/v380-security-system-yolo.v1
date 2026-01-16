#!/usr/bin/env python3
"""Fixed split frame detection implementation."""

print("[1/4] Reading detectors.py...", flush=True)
with open('detectors.py', 'r') as f:
    content = f.read()

# Create backup
print("[2/4] Creating backup...", flush=True)
with open('detectors.py.backup2', 'w') as f:
    f.write(content)

# Find where to insert detect_split_frame method
# Find the detect method and insert before it
print("[3/4] Adding split frame detection method...", flush=True)

detect_method_start = content.find('    def detect(self, frame: np.ndarray, draw_skeleton: bool = False)')

if detect_method_start == -1:
    print("❌ Could not find detect() method!", flush=True)
    exit(1)

# Insert detect_split_frame before detect method
split_method = '''    def detect_split_frame(self, frame, top_conf=0.25, bottom_conf=0.15):
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

content = content[:detect_method_start] + split_method + content[detect_method_start:]

# Update detect() method to use split frame
print("[4/4] Updating detect() method...", flush=True)

# Find the start of detection logic in detect method
detect_logic_start = content.find('persons = []\n        output = frame.copy()', detect_method_start)
if detect_logic_start == -1:
    detect_logic_start = content.find('persons = []', detect_method_start)

if detect_logic_start == -1:
    print("⚠️  Could not find detect logic, keeping original", flush=True)
else:
    # Insert split frame check at the beginning of detection logic
    split_check = '''        # Check if frame is split (1280x720 resolution)
        h, w = frame.shape[:2]
        
        # Use split frame detection for 1280x720 resolution
        if w == 1280 and h == 720:
            try:
                persons = self.detect_split_frame(frame)
                output = frame.copy()
            except Exception as e:
                print(f"[Detector] Split frame error: {e}")
                persons = []
                output = frame.copy()
        else:
        
'''
    
    content = content[:detect_logic_start] + split_check + '            ' + content[detect_logic_start:]

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
print("   1. Added detect_split_frame() method", flush=True)
print("   2. Updated detect() to auto-detect split frame", flush=True)
print("   3. Top camera: confidence 0.25", flush=True)
print("   4. Bottom camera: confidence 0.15 (more sensitive)", flush=True)
print()
print("📋 Next steps:", flush=True)
print("   1. sudo systemctl restart security-system-web", flush=True)
print("   2. sudo systemctl status security-system-web", flush=True)
print("   3. Test at: http://10.26.27.104:8080/web.html", flush=True)
print()
print("💾 Backup created: detectors.py.backup2", flush=True)
print()
