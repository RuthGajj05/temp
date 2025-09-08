#!/usr/bin/env python3
"""
Navigation Agent for PicarX Robot
Handles distance-based navigation tasks like "navigate until 30cm from the couch"
"""

import sys
import os
import time
from typing import Optional

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from primitives import *
from openai import OpenAI
from keys import OPENAI_API_KEY
import base64

class NavigationAgent:
    def __init__(self):
        """Initialize the Navigation Agent."""
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.target_distance = None
        self.target_object = None
        self.navigation_active = False
        self.attempt_count = 0
        
        # Target lock-on with hysteresis
        self.locked_target = None
        self.lock_confidence = 0.0
        self.lock_cycles = 0
        self.max_lock_cycles = 5  # Keep target for 5 cycles once locked
        self.consecutive_consistent_frames = 0  # Track consistent frames for position updates
        self.last_position = None  # Track last position for consistency check
        self.keys_lock = False  # Special lock for "keys" target
        
    def analyze_surroundings(self, target_object: str, estimate_distance: bool = False, panoramic: bool = False) -> str:
        """Analyze the surroundings using 5-frame panorama with single vision call."""
        try:
            # Initialize camera
            camera_result = init_camera()
            print(f"Camera: {camera_result}")
            
            if panoramic:
                # Take 5-frame panorama: left, right, down, center, up
                print("📸 Taking 5-frame panorama for comprehensive analysis...")
                camera_positions = [
                    {"pan": -20, "tilt": 0, "name": "left"},
                    {"pan": 20, "tilt": 0, "name": "right"}, 
                    {"pan": 0, "tilt": -10, "name": "down"},
                    {"pan": 0, "tilt": 0, "name": "center"},
                    {"pan": 0, "tilt": 10, "name": "up"}
                ]
                
                image_paths = []
                for i, pos in enumerate(camera_positions):
                    # Move camera to position
                    pan_result = set_cam_pan_angle(pos["pan"])
                    tilt_result = set_cam_tilt_angle(pos["tilt"])
                    print(f"📷 Camera position {pos['name']}: pan={pos['pan']}°, tilt={pos['tilt']}°")
                    time.sleep(0.3)  # Wait for camera to settle
                    
                    # Take photo
                    photo_result = capture_image(f"panorama_{pos['name']}_{i+1}.jpg")
                    print(f"Photo {pos['name']}: {photo_result}")
                    
                    # Extract file path
                    if "successfully" in photo_result.lower():
                        import re
                        path_match = re.search(r': (.+\.jpg)', photo_result)
                        if path_match:
                            image_paths.append(path_match.group(1))
                        else:
                            print(f"⚠️ Could not extract path for {pos['name']} photo")
                            return f"Failed to extract image path for {pos['name']}"
                    else:
                        print(f"⚠️ Failed to capture {pos['name']} photo")
                        return f"Failed to capture {pos['name']} photo"
                
                # Reset camera to center
                set_cam_pan_angle(0)
                set_cam_tilt_angle(0)
                print("📷 Camera reset to center position")
                
                # Send all 5 images in one vision call with strict format
                return self._analyze_panorama_images(image_paths, target_object, estimate_distance)
                
            else:
                # Take a single picture of the surroundings
                photo_result = capture_image("surroundings_analysis.jpg")
                print(f"Photo: {photo_result}")
                
                # Extract file path from result
                if "successfully" in photo_result.lower():
                    import re
                    path_match = re.search(r': (.+\.jpg)', photo_result)
                    if path_match:
                        image_path = path_match.group(1)
                        return self._analyze_single_image(image_path, target_object, estimate_distance)
                    else:
                        return "Could not extract image path from capture result"
                else:
                    return "Failed to capture image for analysis"
                
        except Exception as e:
            return f"Vision analysis error: {str(e)}"
    
    def vision_reply_missing_images(self, text: str) -> bool:
        """Detect if vision call failed due to missing images."""
        t = (text or "").lower()
        phrases = [
            "unable to view your images",
            "can't view your images",
            "can't analyze images",
            "cannot analyze images",
            "i'm unable to view your images",
        ]
        if any(p in t for p in phrases):
            return True
        # Also treat as failure if our structured fields are missing
        return not any(key in t for key in ("TARGET:", "POSITION:", "BEST_VIEW:", "HORIZ_OFFSET:", "DISTANCE:", "CONFIDENCE:"))
    
    def build_mm_contents(self, view_names, image_paths, log=True):
        """Build safe, flat content list with labels and optional downscaling."""
        import os, time, base64
        try:
            import cv2
        except ImportError:
            cv2 = None

        assert len(view_names) == len(image_paths), "views != images"
        parts = [{
            "type": "text",
            "text": "You will receive labeled views. Use them to pick the NEAREST object and tell me the best view."
        }]
        for view, path in zip(view_names, image_paths):
            # wait up to ~200ms for the file to exist and be non-empty
            for _ in range(20):
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    break
                time.sleep(0.01)
            send_path = path
            # optional resize to keep request small and consistent
            if cv2 is not None:
                img = cv2.imread(path)
                if img is not None:
                    import os as _os
                    h, w = img.shape[:2]
                    if max(h, w) > 800:  # simple cap
                        # scale to 640x480-ish while preserving aspect
                        import math
                        scale = 640.0 / max(h, w)
                        resized = cv2.resize(img, (int(w*scale), int(h*scale)))
                        send_path = _os.path.splitext(path)[0] + ".resized.jpg"
                        cv2.imwrite(send_path, resized, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

            with open(send_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            if log:
                print(f"[vision] {view}: {send_path} base64_len={len(b64)}")
            parts.append({"type": "text", "text": f"VIEW: {view}"})
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        return parts
    
    def analyze_panorama_with_retries(self, view_names, image_paths, prompt_text, center_image_path=None):
        """Robust panorama analysis with retry + fallback."""
        # 1) first try (no recapture)
        contents = self.build_mm_contents(view_names, image_paths)
        print(f"parts_count={len(contents)}, image_parts={sum(1 for p in contents if p.get('type')=='image_url')}")
        
        ans = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": [{"type":"text", "text": prompt_text}] + contents}],
            max_tokens=500
        ).choices[0].message.content or ""

        if not self.vision_reply_missing_images(ans):
            return ans, "ok"

        print("[vision] first try says it can't view images; retrying with recapture+downscale")
        # 2) second try: rebuild contents (if your capture function returns new files, re-call it before this)
        contents = self.build_mm_contents(view_names, image_paths)  # if you re-captured, pass the new paths here
        ans2 = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": [{"type":"text", "text": prompt_text}] + contents}],
            max_tokens=500
        ).choices[0].message.content or ""
        if not self.vision_reply_missing_images(ans2):
            return ans2, "ok_retry"

        print("[vision] still failing; falling back to single-center image then sonar sweep")
        # 3) fallback A: single center image (if provided)
        if center_image_path:
            single = self.build_mm_contents(["center"], [center_image_path])
            ans3 = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": [{"type":"text", "text": prompt_text}] + single}],
                max_tokens=500
            ).choices[0].message.content or ""
            if not self.vision_reply_missing_images(ans3):
                return ans3, "ok_single"

        # 4) fallback B: tell caller to perform sonar sweep
        return "TARGET: not visible\nBEST_VIEW: unknown\nCONFIDENCE: 4\nDETAILS: Vision unavailable; do sonar sweep", "fallback_sonar"
    
    def sonar_sweep_pick_heading(self, arc_deg=180, step_deg=15):
        """
        Rotate in place across a full arc using alternating left/right pattern,
        sample median distance at each heading, and return the best direction.
        """
        import math
        samples = []
        current_angle = 0
        max_steps = arc_deg // step_deg
        
        print(f"📡 Starting alternating sonar sweep: {arc_deg}° arc in {step_deg}° steps")
        
        # Alternating pattern: left, right, left, right, etc.
        for i in range(max_steps):
            if i % 2 == 0:  # Even steps: turn left
                turn_in_place_left(step_deg, 20, 0.5)
                current_angle -= step_deg
                direction = "left"
                angle_display = abs(current_angle)
            else:  # Odd steps: turn right
                turn_in_place_right(step_deg, 20, 0.5)
                current_angle += step_deg
                direction = "right"
                angle_display = current_angle
            
            # Take distance reading
            d = self._get_stable_distance(samples=3)
            if d is not None and d > 0:
                samples.append((direction, current_angle, d))
                print(f"📡 {direction.capitalize()} {angle_display}°: {d}cm")
            else:
                print(f"📡 {direction.capitalize()} {angle_display}°: no reading")
        
        # Return to center
        print(f"📡 Returning to center from {current_angle}°...")
        if current_angle > 0:
            # We're to the right, turn left back to center
            for _ in range(abs(current_angle) // step_deg):
                turn_in_place_left(step_deg, 20, 0.5)
        elif current_angle < 0:
            # We're to the left, turn right back to center
            for _ in range(abs(current_angle) // step_deg):
                turn_in_place_right(step_deg, 20, 0.5)
        
        print(f"📡 Sonar sweep complete. {len(samples)} valid readings collected.")
        
        if not samples:
            print("📡 No reliable sonar readings found")
            return None  # nothing reliable
        
        # Pick heading with smallest distance
        direction, angle, distance = min(samples, key=lambda x: x[2])
        print(f"📡 Closest object found: {distance}cm at {abs(angle)}° {direction}")
        
        return (direction, abs(angle))

    def _analyze_panorama_images(self, image_paths: list, target_object: str, estimate_distance: bool) -> str:
        """Analyze 5 panorama images with robust retry and fallback mechanisms."""
        try:
            view_names = ["left", "right", "down", "center", "up"]
            center_image_path = image_paths[3] if len(image_paths) > 3 else None  # center is index 3
            
            # Create strict prompt for ranking nearest object across views
            if estimate_distance:
                prompt_text = f"""Analyze these 5 images taken from different camera angles (left, right, down, center, up) and find the nearest {target_object}.

STRICT OUTPUT FORMAT REQUIRED:

1. RANK the views by proximity to {target_object} (1=closest, 5=farthest):
RANKING: [view_name] [view_name] [view_name] [view_name] [view_name]

2. For the CLOSEST view, provide:
BEST_VIEW: [view_name]
TARGET: [visible/not visible]
POSITION: [left/right/center] [foreground/middle/background] [size%]
DISTANCE: [specific cm estimate]
CONFIDENCE: [1-10]
DETAILS: [what you see in the closest view]

3. For ALL views, provide brief analysis:
LEFT: [target visible? position? distance?]
RIGHT: [target visible? position? distance?]
DOWN: [target visible? position? distance?]
CENTER: [target visible? position? distance?]
UP: [target visible? position? distance?]

Be CONSERVATIVE with distance estimates. If uncertain, estimate on the FAR side (safer)."""
            else:
                prompt_text = f"""Analyze these 5 images taken from different camera angles (left, right, down, center, up) and find the nearest {target_object}.

STRICT OUTPUT FORMAT REQUIRED:

1. RANK the views by proximity to {target_object} (1=closest, 5=farthest):
RANKING: [view_name] [view_name] [view_name] [view_name] [view_name]

2. For the CLOSEST view, provide:
BEST_VIEW: [view_name]
TARGET: [visible/not visible]
POSITION: [left/right/center] [foreground/middle/background] [size%]
CONFIDENCE: [1-10]
DETAILS: [what you see in the closest view]

3. For ALL views, provide brief analysis:
LEFT: [target visible? position?]
RIGHT: [target visible? position?]
DOWN: [target visible? position?]
CENTER: [target visible? position?]
UP: [target visible? position?]"""
            
            # Use robust panorama analysis with retries
            analysis, status = self.analyze_panorama_with_retries(view_names, image_paths, prompt_text, center_image_path)
            
            if status == "fallback_sonar":
                print("🔄 Vision failed, performing sonar sweep...")
                sweep_result = self.sonar_sweep_pick_heading()
                if sweep_result:
                    side, degrees = sweep_result
                    print(f"📡 Sonar sweep found closest object {degrees}° to the {side}")
                    # Turn toward the closest object
                    if side == "left":
                        turn_in_place_left(degrees, 25, 1.0)
                    else:
                        turn_in_place_right(degrees, 25, 1.0)
                    return f"TARGET: detected by sonar\nBEST_VIEW: {side}\nPOSITION: {side} foreground\nCONFIDENCE: 6\nDETAILS: Target located via sonar sweep at {degrees}° {side}"
                else:
                    return "TARGET: not visible\nBEST_VIEW: unknown\nCONFIDENCE: 2\nDETAILS: No reliable sonar readings found"
            
            print(f"Panorama Analysis ({status}): {analysis}")
            return analysis
            
        except Exception as e:
            print(f"Panorama analysis error: {str(e)}")
            return f"Panorama analysis error: {str(e)}"
    
    def _analyze_single_image(self, image_path: str, target_object: str, estimate_distance: bool) -> str:
        """Analyze a single image with the standard format."""
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            if estimate_distance:
                prompt_text = f"""Look at this image carefully and analyze it step by step:

1. TARGET IDENTIFICATION: Can you see a {target_object}? Be very specific about what you see.

2. POSITION ANALYSIS: If you can see the {target_object}, describe its exact position:
   - Is it on the LEFT side, RIGHT side, or CENTER of the image?
   - Is it in the foreground, middle ground, or background?
   - What percentage of the image does it occupy? (small=5-15%, medium=15-30%, large=30%+)

3. DISTANCE ESTIMATION: Estimate the distance in centimeters based on:
   - Object size in the image (larger = closer)
   - Perspective and depth cues
   - Reference objects of known size
   - Be CONSERVATIVE and specific: "approximately 45cm", "very close at 15-20cm", "far away at 80-100cm"
   - If uncertain, estimate on the FAR side (safer)

4. CONFIDENCE LEVEL: Rate your confidence (1-10) for both position and distance estimates.

5. ALTERNATIVES: If you cannot see a {target_object}, describe what objects you DO see and their positions.

Format your response as:
TARGET: [visible/not visible]
POSITION: [left/right/center] [foreground/middle/background] [size%]
DISTANCE: [specific cm estimate]
CONFIDENCE: [1-10]
DETAILS: [additional observations]"""
            else:
                prompt_text = f"""Look at this image carefully and analyze it step by step:

1. TARGET IDENTIFICATION: Can you see a {target_object}? Be very specific about what you see.

2. POSITION ANALYSIS: If you can see the {target_object}, describe its exact position:
   - Is it on the LEFT side, RIGHT side, or CENTER of the image?
   - Is it in the foreground, middle ground, or background?
   - What percentage of the image does it occupy? (small=5-15%, medium=15-30%, large=30%+)

3. CONFIDENCE LEVEL: Rate your confidence (1-10) for the position estimate.

4. ALTERNATIVES: If you cannot see a {target_object}, describe what objects you DO see and their positions.

Format your response as:
TARGET: [visible/not visible]
POSITION: [left/right/center] [foreground/middle/background] [size%]
CONFIDENCE: [1-10]
DETAILS: [additional observations]"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt_text
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            analysis = response.choices[0].message.content
            print(f"Single Image Analysis: {analysis}")
            return analysis
            
        except Exception as e:
            return f"Single image analysis error: {str(e)}"
    
    def _get_stable_distance(self, samples: int = 5, near_field: bool = False) -> Optional[float]:
        """Get a stable distance reading using median filtering to avoid spikes and invalid readings."""
        import re
        
        # Adjust samples for near-field behavior
        if near_field:
            samples = 7  # More samples for near-field
            print("📡 Near-field mode: Using 7 samples with motor stop")
        
        vals = []
        for i in range(samples):
            try:
                # Stop motors before reading for better accuracy in near-field
                if near_field and i == 0:
                    stop()  # Stop motors before first reading
                    time.sleep(0.1)  # Brief pause for motor stop
                
                distance_result = get_ultrasonic_distance()  # returns a string like "... at XX cm"
                # Use regex to extract the number from the string
                match = re.search(r'([-+]?\d+(?:\.\d+)?)\s*cm', distance_result)
                if match:
                    d = float(match.group(1))
                    if 2.0 <= d <= 400.0:  # accept only plausible echoes
                        vals.append(d)
                        print(f"📡 Sample {i+1}: {d}cm")
                    else:
                        print(f"📡 Sample {i+1}: {d}cm (invalid range)")
                else:
                    print(f"📡 Sample {i+1}: No valid number found in '{distance_result}'")
            except Exception as e:
                print(f"📡 Sample {i+1}: Error - {e}")
            
            if i < samples - 1:
                time.sleep(0.06)  # Short wait between samples
        
        if not vals:
            print("📡 No valid ultrasonic readings obtained")
            return None
        
        # Return median value for stability
        vals.sort()
        median_val = vals[len(vals)//2]
        print(f"📡 Ultrasonic median: {median_val}cm (from {len(vals)} valid samples)")
        return median_val
    
    def _extract_distance_from_vision(self, vision_analysis: str) -> Optional[float]:
        """Extract distance estimate from vision analysis text with conservative validation."""
        try:
            import re
            
            # First try to read from the DISTANCE line specifically
            distance_line = re.search(r'DISTANCE:\s*([^\n]+)', vision_analysis, re.IGNORECASE)
            text = distance_line.group(1) if distance_line else vision_analysis
            
            # Look for distance patterns in the text (now supports decimals)
            distance_patterns = [
                r'(\d+(?:\.\d+)?)\s*cm',  # "50cm", "30.5 cm"
                r'about\s+(\d+(?:\.\d+)?)\s*cm',  # "about 50cm"
                r'approximately\s+(\d+(?:\.\d+)?)\s*cm',  # "approximately 50cm"
                r'(\d+(?:\.\d+)?)\s*centimeters',  # "50 centimeters"
                r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*cm',  # "10-15cm" (take average)
                r'(\d+(?:\.\d+)?)\s*to\s*(\d+(?:\.\d+)?)\s*cm',  # "10 to 15cm" (take average)
            ]
            
            for pattern in distance_patterns:
                matches = re.findall(pattern, text.lower())
                if matches:
                    if len(matches[0]) == 2:  # Range like "10-15cm"
                        try:
                            min_dist = float(matches[0][0])
                            max_dist = float(matches[0][1])
                            distance = (min_dist + max_dist) / 2  # Return average
                            
                            # Softer validation for camera estimates - accept wider range
                            if 2 <= distance <= 500:  # Much wider range for camera
                                return distance
                            else:
                                print(f"⚠️ Camera distance {distance}cm outside very wide range, using fallback")
                                continue
                        except ValueError:
                            continue
                    else:  # Single distance
                        try:
                            distance = float(matches[0])
                            
                            # Softer validation for camera estimates - accept wider range
                            if 2 <= distance <= 500:  # Much wider range for camera
                                return distance
                            else:
                                print(f"⚠️ Camera distance {distance}cm outside very wide range, using fallback")
                                continue
                        except ValueError:
                            continue
            
            # Look for qualitative distance descriptions (more conservative)
            if any(word in vision_analysis.lower() for word in ['very close', 'extremely close', 'right next to', 'touching']):
                return 15.0  # Conservative very close
            elif any(word in vision_analysis.lower() for word in ['close', 'near', 'nearby']):
                return 40.0  # Conservative close
            elif any(word in vision_analysis.lower() for word in ['medium distance', 'moderate distance']):
                return 80.0  # Conservative medium distance
            elif any(word in vision_analysis.lower() for word in ['far', 'distant', 'far away']):
                return 150.0  # Conservative far
            
            return None
            
        except Exception as e:
            print(f"Error extracting distance from vision: {e}")
            return None
    
    def _determine_movement_strategy(self, vision_analysis: str, current_distance: float, target_distance: float) -> str:
        """Determine movement strategy - just move forward toward the target."""
        try:
            # Use short, safe steps with ultrasonic sensor for distance
            if current_distance is None or current_distance < 0 or current_distance > 500:
                # Invalid sonar - use very small, cautious steps
                step_size = 0.2
                speed = 15
                print(f"⚠️ Invalid sonar - using very cautious steps (speed: {speed}%, duration: {step_size}s)")
            elif current_distance > target_distance + 30:  # Far from target
                step_size = 0.5
                speed = 25
            elif current_distance > target_distance + 15:  # Medium distance
                step_size = 0.3
                speed = 20
            else:  # Close to target
                step_size = 0.2
                speed = 15
            
            # Move forward with short, safe steps
            print(f"🎯 Moving forward toward target (speed: {speed}%, duration: {step_size}s)")
            return move_forward(speed, step_size, check_obstacles=True)
                    
        except Exception as e:
            print(f"Error in movement strategy: {e}")
            # Fallback to simple forward movement
            return move_forward(20, 0.3, check_obstacles=True)
    
    def _extract_target_position(self, vision_analysis: str) -> str:
        """Extract target position (left, right, center) from structured vision analysis."""
        try:
            import re
            
            # Look for structured position format: "POSITION: [left/right/center]"
            position_match = re.search(r'POSITION:\s*([^\\n]+)', vision_analysis, re.IGNORECASE)
            if position_match:
                position_text = position_match.group(1).lower()
                if "left" in position_text:
                    return "left"
                elif "right" in position_text:
                    return "right"
                elif "center" in position_text or "middle" in position_text:
                    return "center"
            
            # Fallback to old method if structured format not found
            if any(word in vision_analysis.lower() for word in ["left", "left side", "to the left"]):
                return "left"
            elif any(word in vision_analysis.lower() for word in ["right", "right side", "to the right"]):
                return "right"
            elif any(word in vision_analysis.lower() for word in ["center", "middle", "straight ahead", "directly ahead"]):
                return "center"
            else:
                return "unknown"
        except Exception as e:
            print(f"Error extracting target position: {e}")
            return "unknown"
    
    def _extract_confidence_level(self, vision_analysis: str) -> float:
        """Extract confidence level from structured vision analysis."""
        try:
            import re
            
            # Look for confidence format: "CONFIDENCE: [1-10]"
            confidence_match = re.search(r'CONFIDENCE:\s*(\d+)', vision_analysis, re.IGNORECASE)
            if confidence_match:
                return float(confidence_match.group(1))
            
            return 5.0  # Default medium confidence
        except Exception as e:
            print(f"Error extracting confidence level: {e}")
            return 5.0
    
    def _extract_best_view_and_direction(self, vision_analysis: str) -> tuple[str, str]:
        """Extract the best view and direction from panorama analysis."""
        try:
            import re
            
            # Look for BEST_VIEW line
            best_view_match = re.search(r'BEST_VIEW:\s*([^\n]+)', vision_analysis, re.IGNORECASE)
            if best_view_match:
                best_view = best_view_match.group(1).strip().lower()
                
                # Map view to direction
                if best_view == "left":
                    return "left", "left"
                elif best_view == "right":
                    return "right", "right"
                elif best_view == "center":
                    return "center", "center"
                elif best_view == "down":
                    return "down", "center"  # Down view, move forward
                elif best_view == "up":
                    return "up", "center"    # Up view, move forward
                else:
                    return best_view, "center"
            
            # Fallback: look for RANKING line and take first
            ranking_match = re.search(r'RANKING:\s*([^\n]+)', vision_analysis, re.IGNORECASE)
            if ranking_match:
                ranking = ranking_match.group(1).strip().lower()
                first_view = ranking.split()[0] if ranking.split() else "center"
                
                if first_view == "left":
                    return "left", "left"
                elif first_view == "right":
                    return "right", "right"
                else:
                    return first_view, "center"
            
            return "center", "center"
            
        except Exception as e:
            print(f"Error extracting best view: {e}")
            return "center", "center"
    
    def _turn_until_centered(self, target_object: str, initial_direction: str) -> None:
        """Keep turning until the target object is in the center of vision."""
        try:
            max_attempts = 8  # Prevent infinite turning
            attempt = 0
            
            while attempt < max_attempts:
                # Take a single photo to check current position
                photo_result = capture_image(f"centering_check_{attempt}.jpg")
                print(f"📸 Centering check photo: {photo_result}")
                
                if "successfully" in photo_result.lower():
                    # Analyze the current view
                    vision_analysis = self.analyze_surroundings(target_object, estimate_distance=False, panoramic=False)
                    print(f"Centering Analysis: {vision_analysis}")
                    
                    # Check if target is now centered
                    target_position = self._extract_target_position(vision_analysis.lower())
                    confidence = self._extract_confidence_level(vision_analysis)
                    
                    print(f"🎯 Current position: {target_position}, confidence: {confidence}")
                    
                    if target_position == "center" and confidence >= 6:
                        print("✅ Target is now centered!")
                        return
                    elif target_position == "left" or (initial_direction == "left" and target_position != "center"):
                        print("🔄 Target still on left, turning left...")
                        turn_result = turn_in_place_left(20, 20, 0.8)
                        print(f"Turn result: {turn_result}")
                    elif target_position == "right" or (initial_direction == "right" and target_position != "center"):
                        print("🔄 Target still on right, turning right...")
                        turn_result = turn_in_place_right(20, 20, 0.8)
                        print(f"Turn result: {turn_result}")
                    else:
                        print("🔄 Target position unclear, making small adjustment...")
                        if initial_direction == "left":
                            turn_result = turn_in_place_left(15, 15, 0.6)
                        else:
                            turn_result = turn_in_place_right(15, 15, 0.6)
                        print(f"Turn result: {turn_result}")
                    
                    attempt += 1
                    time.sleep(0.5)  # Brief pause between attempts
                else:
                    print("⚠️ Failed to capture centering photo, making small turn...")
                    if initial_direction == "left":
                        turn_result = turn_in_place_left(15, 15, 0.6)
                    else:
                        turn_result = turn_in_place_right(15, 15, 0.6)
                    print(f"Turn result: {turn_result}")
                    attempt += 1
                    time.sleep(0.5)
            
            print("⚠️ Reached max centering attempts, proceeding with current orientation")
            
        except Exception as e:
            print(f"Error in centering: {e}")
            print("⚠️ Centering failed, proceeding with current orientation")
    
    def _extract_target_from_vision(self, vision_analysis: str) -> str:
        """Extract what the robot actually sees as the target object."""
        try:
            import re
            
            # Look for TARGET line in structured format
            target_match = re.search(r'TARGET:\s*([^\n]+)', vision_analysis, re.IGNORECASE)
            if target_match:
                target_status = target_match.group(1).strip()
                if "visible" in target_status.lower():
                    # Look for DETAILS section to get more specific info
                    details_match = re.search(r'DETAILS:\s*([^\n]+)', vision_analysis, re.IGNORECASE)
                    if details_match:
                        details = details_match.group(1).strip()
                        # Extract object description from details
                        if "nearest object" in details.lower():
                            return f"Nearest object detected: {details}"
                        elif "small item" in details.lower():
                            return f"Small item detected: {details}"
                        elif "coasters" in details.lower():
                            return f"Coasters detected: {details}"
                        else:
                            return f"Object detected: {details}"
                    else:
                        return f"Target visible: {target_status}"
                else:
                    return f"Target not visible: {target_status}"
            
            # Fallback: look for any object mentions in the analysis
            if "nearest object" in vision_analysis.lower():
                return "Nearest object (from analysis)"
            elif "small item" in vision_analysis.lower():
                return "Small item (from analysis)"
            elif "coasters" in vision_analysis.lower():
                return "Coasters (from analysis)"
            else:
                return "Unknown object (from analysis)"
                
        except Exception as e:
            print(f"Error extracting target from vision: {e}")
            return "Error parsing target"
    
    def _update_target_lock(self, confidence: float, target_position: str, vision_analysis: str) -> tuple[bool, str]:
        """
        Update target lock-on with intelligent hysteresis logic.
        
        Features:
        - Lock updates sides based on higher confidence or consistency
        - Keys lock promotion with higher hysteresis threshold
        - Vision for direction, sonar for distance
        - Softer range validation for camera estimates
        
        Returns:
            (is_locked, reason) - whether target is locked and why
        """
        try:
            # Check if target is visible in current analysis
            vision_lower = vision_analysis.lower()
            target_visible = "visible" in vision_lower and "not visible" not in vision_lower
            
            # Check for "keys" target promotion
            keys_mentioned = "keys" in vision_lower and confidence >= 7.0
            if keys_mentioned and not self.keys_lock:
                self.keys_lock = True
                print("🔑 Keys target detected - promoting to special lock")
            
            # If we have a locked target, check if we should keep it or update it
            if self.locked_target is not None:
                self.lock_cycles += 1
                
                # Check for position updates (higher confidence or consistency)
                position_changed = target_position != self.locked_target.get('position', 'unknown')
                higher_confidence = confidence >= (self.lock_confidence + 1.0)
                
                # Check for consistent frames
                if target_position == self.last_position:
                    self.consecutive_consistent_frames += 1
                else:
                    self.consecutive_consistent_frames = 0
                
                consistent_frames = self.consecutive_consistent_frames >= 2
                
                # Update lock position if conditions are met
                if target_visible and position_changed and (higher_confidence or consistent_frames):
                    old_position = self.locked_target.get('position', 'unknown')
                    self.locked_target['position'] = target_position
                    self.locked_target['confidence'] = max(self.lock_confidence, confidence)
                    self.lock_confidence = self.locked_target['confidence']
                    print(f"🔄 Lock position updated: {old_position} → {target_position}")
                
                # Check unlock conditions (with higher threshold for keys)
                unlock_threshold = 3.0 if self.keys_lock else 2.0
                
                if not target_visible:
                    # Target disappeared - unlock
                    self.locked_target = None
                    self.lock_cycles = 0
                    self.consecutive_consistent_frames = 0
                    self.keys_lock = False
                    return False, "Target disappeared"
                
                if confidence < (self.lock_confidence - unlock_threshold):
                    # Confidence dropped significantly - unlock
                    self.locked_target = None
                    self.lock_cycles = 0
                    self.consecutive_consistent_frames = 0
                    self.keys_lock = False
                    return False, f"Confidence dropped from {self.lock_confidence:.1f} to {confidence:.1f} (threshold: {unlock_threshold})"
                
                if self.lock_cycles >= self.max_lock_cycles:
                    # Lock expired - unlock
                    self.locked_target = None
                    self.lock_cycles = 0
                    self.consecutive_consistent_frames = 0
                    self.keys_lock = False
                    return False, f"Lock expired after {self.max_lock_cycles} cycles"
                
                # Still locked - update confidence
                self.lock_confidence = max(self.lock_confidence, confidence)
                self.last_position = target_position
                return True, "Target remains locked"
            
            # No locked target - check if we should lock on
            if confidence >= 7.0 and target_visible and target_position != "unknown":
                # High confidence target detected - lock on
                self.locked_target = {
                    'position': target_position,
                    'confidence': confidence,
                    'analysis': vision_analysis
                }
                self.lock_confidence = confidence
                self.lock_cycles = 0
                self.consecutive_consistent_frames = 0
                self.last_position = target_position
                lock_reason = f"Target locked with confidence {confidence:.1f}"
                if self.keys_lock:
                    lock_reason += " (keys target - special lock)"
                return True, lock_reason
            
            # Not enough confidence to lock
            self.last_position = target_position
            return False, f"Confidence {confidence:.1f} too low for lock (need ≥7.0)"
            
        except Exception as e:
            print(f"Error in target lock update: {e}")
            return False, f"Lock error: {str(e)}"
    

    def navigate_to_distance(self, target_distance: float, target_object: str = "object") -> str:
        """
        Navigate towards the nearest object until reaching the specified distance.
        
        Args:
            target_distance: Distance in cm to stop from the object
            target_object: Description of what we're navigating to (for context)
        """
        try:
            self.target_distance = target_distance
            self.target_object = target_object
            self.navigation_active = True
            
            # Reset robot to starting position
            reset_result = reset()
            print(f"Reset: {reset_result}")
            
            # Reset target lock for new navigation task
            self.locked_target = None
            self.lock_confidence = 0.0
            self.lock_cycles = 0
            self.consecutive_consistent_frames = 0
            self.last_position = None
            self.keys_lock = False
            print("🔓 Target lock reset for new navigation task")
            
            # Analyze surroundings using 5-frame panorama
            print(f"🎯 TARGET DESIGNATION: Looking for '{target_object}'")
            print(f"🔍 Taking 5-frame panorama to find {target_object}...")
            vision_analysis = self.analyze_surroundings(target_object, estimate_distance=True, panoramic=True)
            print(f"Panorama Analysis: {vision_analysis}")
            
            # Extract best view and direction from panorama
            best_view, target_direction = self._extract_best_view_and_direction(vision_analysis)
            print(f"🎯 BEST VIEW: {best_view} -> Direction: {target_direction}")
            
            # Keep turning until the target is in the center of vision
            if target_direction != "center":
                print(f"🔄 Target is {target_direction}, turning until it's centered...")
                self._turn_until_centered(target_object, target_direction)
            else:
                print("✅ Target is already centered, proceeding forward...")
            
            # Wait a moment for system to stabilize
            print("⏳ Stabilizing sensors...")
            time.sleep(1.0)
            
            # Start navigation loop
            invalid_readings = 0
            max_invalid_readings = 8  # Increased - try ultrasonic more before camera
            consecutive_vision_attempts = 0
            max_vision_attempts = 2  # Reduced - use camera less often
            
            while self.navigation_active:
                # Determine if we're in near-field mode (< 40cm)
                near_field_mode = False
                if self.locked_target is not None:
                    # Use locked target distance if available
                    locked_distance = self._extract_distance_from_vision(self.locked_target.get('analysis', ''))
                    if locked_distance is not None and locked_distance < 40:
                        near_field_mode = True
                
                # Get current distance with appropriate sampling
                current_distance = self._get_stable_distance(near_field=near_field_mode)
                
                # Check if we got a valid distance reading
                if current_distance is None or current_distance < 0 or current_distance > 500:
                    # Invalid reading (None, negative, or too large) - use camera immediately
                    consecutive_vision_attempts += 1
                    print(f"⚠️ Invalid ultrasonic reading ({current_distance}), using camera backup (attempt {consecutive_vision_attempts})")
                    
                    if consecutive_vision_attempts >= max_vision_attempts:
                        # Instead of failing, stick to camera-only mode for a while
                        print("🔄 Switching to camera-only mode due to persistent ultrasonic issues")
                        consecutive_vision_attempts = 0  # Reset counter to continue with camera
                        time.sleep(0.5)  # Brief pause before continuing
                    
                    # Use camera as backup for distance estimation
                    print("📷 Using camera for distance estimation...")
                    vision_analysis = self.analyze_surroundings(target_object, estimate_distance=True)
                    print(f"Vision Analysis: {vision_analysis}")
                    
                    # Extract distance from vision with conservative approach
                    vision_distance = self._extract_distance_from_vision(vision_analysis)
                    if vision_distance is not None:
                        # Use conservative camera distance with safety margin
                        current_distance = min(vision_distance * 1.2, 100)  # Add 20% safety margin, cap at 100cm
                        print(f"📷 Camera distance estimate: {vision_distance}cm -> Conservative: {current_distance}cm")
                        invalid_readings = 0  # Reset invalid counter since we got a reading
                        consecutive_vision_attempts = 0  # Reset camera backup counter since camera succeeded
                    else:
                        print("⚠️ Could not extract distance from vision analysis, using safe fallback")
                        current_distance = 60.0  # Safe fallback distance
                        print(f"📷 Using safe fallback distance: {current_distance}cm")
                else:
                    # Valid ultrasonic reading, reset counters
                    invalid_readings = 0
                    consecutive_vision_attempts = 0
                    print(f"📡 Ultrasonic distance: {current_distance}cm")
                
                # Skip obstacle detection if we're using camera backup (invalid ultrasonic reading)
                if current_distance is not None and current_distance >= 0 and current_distance <= 500:
                    # Only check for obstacles if we have a valid ultrasonic reading
                    if current_distance < 5.0:
                        stop_result = stop()
                        self.navigation_active = False
                        return f"⚠️ Safety stop! Too close to {target_object} at {current_distance:.1f}cm. {stop_result}"
                
                # Check if we've reached the target distance
                if current_distance <= target_distance:
                    stop_result = stop()
                    self.navigation_active = False
                    
                    # Get final target identification
                    final_target = self._extract_target_from_vision(vision_analysis)
                    return f"✅ Navigation complete! Reached {target_distance}cm from {target_object}. Final distance: {current_distance:.1f}cm. Target identified as: {final_target}. {stop_result}"
                
                # Check if we're too close (safety check)
                if current_distance < 5.0:
                    stop_result = stop()
                    self.navigation_active = False
                    return f"⚠️ Safety stop! Too close to {target_object} at {current_distance:.1f}cm. {stop_result}"
                
                # Determine movement strategy based on vision analysis
                movement_result = self._determine_movement_strategy(vision_analysis, current_distance, target_distance)
                print(f"Movement: {movement_result}")
                
                # Check if movement was stopped due to obstacles
                if "stopped early" in movement_result.lower():
                    self.navigation_active = False
                    return f"❌ Navigation stopped due to obstacle detection: {movement_result}"
                
                # Take photo every 2 attempts for consistent monitoring
                if self.attempt_count % 2 == 0:
                    photo_result = capture_image(f"navigation_step_{self.attempt_count}.jpg")
                    print(f"📸 Step photo: {photo_result}")
                
                # Periodically re-analyze surroundings with single image (more efficient)
                if self.attempt_count % 5 == 0:  # Every 5 attempts (less frequent for efficiency)
                    print("🔄 Re-analyzing surroundings with single image...")
                    vision_analysis = self.analyze_surroundings(target_object, estimate_distance=True, panoramic=False)
                    print(f"Updated Vision Analysis: {vision_analysis}")
                    
                    # Cross-check with vision distance if ultrasonic is working
                    if current_distance is not None:
                        vision_distance = self._extract_distance_from_vision(vision_analysis)
                        if vision_distance is not None:
                            print(f"📷 Vision distance: {vision_distance}cm vs 📊 Current estimate: {current_distance}cm")
                            # Use average of both if they're reasonably close
                            if abs(vision_distance - current_distance) < 30:  # Within 30cm
                                current_distance = (current_distance + vision_distance) / 2
                                print(f"📊 Using combined distance: {current_distance}cm")
                
                self.attempt_count += 1
                # Small delay for sensor stability
                time.sleep(0.1)
            
            return "Navigation completed successfully"
            
        except Exception as e:
            self.navigation_active = False
            return f"Navigation error: {str(e)}"
    
    def stop_navigation(self) -> str:
        """Stop the current navigation task."""
        self.navigation_active = False
        stop_result = stop()
        return f"Navigation stopped. {stop_result}"
    
    def get_navigation_status(self) -> str:
        """Get current navigation status."""
        if not self.navigation_active:
            return "No active navigation task"
        
        try:
            distance_result = get_ultrasonic_distance()
            status = f"Navigating to {self.target_object} (target: {self.target_distance}cm). Current: {distance_result}"
            
            # Add target lock status
            if self.locked_target is not None:
                lock_type = "🔑 Keys lock" if self.keys_lock else "🔒 Target locked"
                status += f"\n{lock_type}: {self.locked_target['position']} (confidence: {self.lock_confidence:.1f}, cycles: {self.lock_cycles}/{self.max_lock_cycles})"
                status += f"\nConsistent frames: {self.consecutive_consistent_frames}"
            else:
                status += "\n🔓 No target lock"
            
            return status
        except Exception as e:
            return f"Status check error: {str(e)}"

def main():
    """Interactive navigation agent."""
    print("🤖 PicarX Navigation Agent")
    print("=" * 40)
    print("Commands:")
    print("- 'navigate to 30cm from couch' - Navigate until 30cm from nearest object")
    print("- 'status' - Check current navigation status")
    print("- 'stop' - Stop current navigation")
    print("- 'quit' - Exit")
    print()
    
    agent = NavigationAgent()
    
    while True:
        try:
            command = input("Navigation> ").strip().lower()
            
            if command == "quit":
                print("👋 Goodbye!")
                break
            elif command == "status":
                result = agent.get_navigation_status()
                print(f"Status: {result}")
            elif command == "stop":
                result = agent.stop_navigation()
                print(f"Stop: {result}")
            elif command.startswith("navigate to"):
                # Parse command like "navigate to 30cm from couch"
                try:
                    parts = command.split()
                    distance_idx = -1
                    for i, part in enumerate(parts):
                        if part.endswith("cm"):
                            distance_idx = i
                            break
                    
                    if distance_idx == -1:
                        print("❌ Please specify distance in cm (e.g., 'navigate to 30cm from couch')")
                        continue
                    
                    distance = float(parts[distance_idx].replace("cm", ""))
                    object_name = "object"
                    
                    # Try to extract object name
                    if "from" in parts:
                        from_idx = parts.index("from")
                        if from_idx + 1 < len(parts):
                            object_name = " ".join(parts[from_idx + 1:])
                    
                    print(f"🚗 Starting navigation to {distance}cm from {object_name}...")
                    result = agent.navigate_to_distance(distance, object_name)
                    print(f"Result: {result}")
                    
                except (ValueError, IndexError) as e:
                    print(f"❌ Error parsing command: {e}")
                    print("Format: 'navigate to 30cm from couch'")
            else:
                print("❌ Unknown command. Type 'quit' to exit.")
                
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
            agent.stop_navigation()
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
