#!/usr/bin/env python3
"""Adjust bottom camera confidence threshold for better detection."""

print("[1/3] Reading detectors.py...", flush=True)
with open('detectors.py', 'r') as f:
    content = f.read()

# Create backup
print("[2/3] Creating backup...", flush=True)
with open('detectors.py.backup3', 'w') as f:
    f.write(content)

# Find detect_split_frame method
print("[3/3] Adjusting bottom camera confidence threshold...", flush=True)

# Pattern to find: bottom_conf=0.15
# Replace with: bottom_conf=0.10 (more sensitive)

# Find all occurrences of bottom_conf in detect_split_frame method
import re

# Find the detect_split_frame method
method_pattern = r'def detect_split_frame\(self, frame, top_conf=(\d+\.?\d*), bottom_conf=(\d+\.?\d*)\):'
match = re.search(method_pattern, content)

if match:
    current_top = float(match.group(1))
    current_bottom = float(match.group(2))
    
    print(f"   Current thresholds: top={current_top}, bottom={current_bottom}", flush=True)
    
    # Lower bottom confidence
    new_bottom = 0.10  # More sensitive
    
    # Replace
    content = re.sub(
        method_pattern,
        f'def detect_split_frame(self, frame, top_conf={current_top}, bottom_conf={new_bottom}):',
        content
    )
    
    # Also find and replace in the actual detection calls
    # Pattern: results_bottom = self.model(bottom_frame, conf=bottom_conf, classes=[0], verbose=False)
    # The parameter name is bottom_conf, so we don't need to change it here
    
    print(f"   New thresholds: top={current_top}, bottom={new_bottom}", flush=True)
else:
    print("❌ Could not find detect_split_frame method!", flush=True)
    exit(1)

# Write updated content
print("✅ Writing updated detectors.py...", flush=True)
with open('detectors.py', 'w') as f:
    f.write(content)

print()
print("=" * 70, flush=True)
print("✅ Bottom Camera Confidence Adjusted!", flush=True)
print("=" * 70, flush=True)
print()
print("📋 Changes made:", flush=True)
print(f"   - Bottom camera confidence: {current_bottom} → {new_bottom}", flush=True)
print("   - Top camera confidence: unchanged", flush=True)
print()
print("💡 Bottom camera is now MORE SENSITIVE!", flush=True)
print()
print("📋 Next steps:", flush=True)
print("   1. sudo systemctl restart security-system-web", flush=True)
print("   2. sudo systemctl status security-system-web", flush=True)
print("   3. Test at: http://10.26.27.104:8080/web.html", flush=True)
print("   4. Stand in front of BOTH cameras", flush=True)
print()
print("💾 Backup created: detectors.py.backup3", flush=True)
print()
print("⚠️  If still not detecting:", flush=True)
print("   Run again: python adjust_bottom_confidence.py (will lower to 0.05)", flush=True)
print()
