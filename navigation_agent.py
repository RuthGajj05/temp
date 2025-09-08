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
        
    def analyze_surroundings(self, target_object: str, estimate_distance: bool = False, multiple_photos: bool = False) -> str:
        """Analyze the surroundings to identify the target object and optionally estimate distance."""
        try:
            # Initialize camera
            camera_result = init_camera()
            print(f"Camera: {camera_result}")
            
            if multiple_photos:
                # Use camera pan/tilt to take photos from different angles
                print("📸 Taking panoramic photos for better spatial analysis...")
                photo_results = []
                analysis_results = []
                
                # Define camera positions for panoramic view
                camera_positions = [
                    {"pan": -20, "tilt": 0, "name": "left"},
                    {"pan": 0, "tilt": 0, "name": "center"}, 
                    {"pan": 20, "tilt": 0, "name": "right"},
                    {"pan": 0, "tilt": -10, "name": "down"},
                    {"pan": 0, "tilt": 10, "name": "up"}
                ]
                
                for i, pos in enumerate(camera_positions):
                    # Move camera to position
                    pan_result = set_cam_pan_angle(pos["pan"])
                    tilt_result = set_cam_tilt_angle(pos["tilt"])
                    print(f"📷 Camera position {pos['name']}: pan={pos['pan']}°, tilt={pos['tilt']}°")
                    time.sleep(0.3)  # Wait for camera to settle
                    
                    # Take photo
                    photo_result = capture_image(f"panoramic_{pos['name']}_{i+1}.jpg")
                    photo_results.append(photo_result)
                    print(f"Photo {pos['name']}: {photo_result}")
                    
                    # Quick analysis of this specific view
                    if "successfully" in photo_result.lower():
                        quick_analysis = self._quick_analyze_photo(photo_result, target_object, pos["name"])
                        analysis_results.append(quick_analysis)
                        print(f"Quick analysis {pos['name']}: {quick_analysis}")
                
                # Reset camera to center
                set_cam_pan_angle(0)
                set_cam_tilt_angle(0)
                print("📷 Camera reset to center position")
                
                # Use the center photo for main analysis
                photo_result = photo_results[1]  # Center photo
                
                # Combine all quick analyses for better context
                combined_analysis = self._combine_panoramic_analysis(analysis_results, target_object)
                print(f"🔍 Combined panoramic analysis: {combined_analysis}")
                
            else:
                # Take a single picture of the surroundings
                photo_result = capture_image("surroundings_analysis.jpg")
                print(f"Photo: {photo_result}")
            
            # Extract file path from result
            if "successfully" in photo_result.lower():
                # Find the file path in the result
                import re
                path_match = re.search(r': (.+\.jpg)', photo_result)
                if path_match:
                    image_path = path_match.group(1)
                    
                    # Create enhanced prompt for better accuracy
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
                    
                    # Analyze the image with GPT-4 Vision
                    with open(image_path, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    
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
                    print(f"Vision Analysis: {analysis}")
                    return analysis
                else:
                    return "Could not extract image path from capture result"
            else:
                return "Failed to capture image for analysis"
                
        except Exception as e:
            return f"Vision analysis error: {str(e)}"
    
    def _get_stable_distance(self, samples: int = 5) -> Optional[float]:
        """Get a stable distance reading using median filtering to avoid spikes and invalid readings."""
        import re
        
        vals = []
        for i in range(samples):
            try:
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
                            
                            # Conservative validation for camera estimates
                            if 5 <= distance <= 200:  # Reasonable range for camera
                                return distance
                            else:
                                print(f"⚠️ Camera distance {distance}cm outside reasonable range, using fallback")
                                continue
                        except ValueError:
                            continue
                    else:  # Single distance
                        try:
                            distance = float(matches[0])
                            
                            # Conservative validation for camera estimates
                            if 5 <= distance <= 200:  # Reasonable range for camera
                                return distance
                            else:
                                print(f"⚠️ Camera distance {distance}cm outside reasonable range, using fallback")
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
        """Determine the best movement strategy based on vision analysis and distance."""
        try:
            vision_lower = vision_analysis.lower()
            
            # Extract confidence level
            confidence = self._extract_confidence_level(vision_analysis)
            print(f"🎯 Vision confidence: {confidence}/10")
            
            # Check if target is visible
            if "not visible" in vision_lower or "cannot see" in vision_lower or confidence < 3:
                print("🎯 Target not visible or low confidence, searching...")
                # Turn to search for the target
                if self.attempt_count % 2 == 0:
                    return turn_in_place_left(30, 25, 1.0)
                else:
                    return turn_in_place_right(30, 25, 1.0)
            
            # Determine target position from vision analysis
            target_position = self._extract_target_position(vision_lower)
            
            # Show what the robot is targeting
            current_target = self._extract_target_from_vision(vision_analysis)
            print(f"🎯 Current target: {current_target}")
            print(f"📍 Target position: {target_position}")
            
            # Adjust movement based on confidence level
            if confidence < 5:
                print("⚠️ Low confidence vision, using conservative movement")
                # Use smaller angles and slower speeds for low confidence
                angle_multiplier = 0.5
                speed_multiplier = 0.7
            elif confidence < 7:
                print("⚠️ Medium confidence vision, using moderate movement")
                # Use normal angles and speeds
                angle_multiplier = 0.8
                speed_multiplier = 0.9
            else:
                print("✅ High confidence vision, using full movement")
                # Use full angles and speeds
                angle_multiplier = 1.0
                speed_multiplier = 1.0
            
            # Choose movement based on target position and distance
            if current_distance > target_distance + 20:  # Far from target
                if target_position == "left":
                    angle = int(30 * angle_multiplier)
                    speed = int(30 * speed_multiplier)
                    print(f"🎯 Target on left, turning in place left then moving forward (angle: {angle}°, speed: {speed}%)")
                    turn_result = turn_in_place_left(angle, speed, 1.0)
                    print(f"Turn result: {turn_result}")
                    return move_forward(speed, 0.8, check_obstacles=True)
                elif target_position == "right":
                    angle = int(30 * angle_multiplier)
                    speed = int(30 * speed_multiplier)
                    print(f"🎯 Target on right, turning in place right then moving forward (angle: {angle}°, speed: {speed}%)")
                    turn_result = turn_in_place_right(angle, speed, 1.0)
                    print(f"Turn result: {turn_result}")
                    return move_forward(speed, 0.8, check_obstacles=True)
                else:  # center or unknown
                    speed = int(30 * speed_multiplier)
                    print(f"🎯 Target ahead, moving forward (speed: {speed}%)")
                    return move_forward(speed, 0.8, check_obstacles=True)
            
            elif current_distance > target_distance + 10:  # Medium distance
                if target_position == "left":
                    angle = int(20 * angle_multiplier)
                    speed = int(25 * speed_multiplier)
                    print(f"🎯 Target on left, gentle turn in place left then moving forward (angle: {angle}°, speed: {speed}%)")
                    turn_result = turn_in_place_left(angle, speed, 0.8)
                    print(f"Turn result: {turn_result}")
                    return move_forward(speed, 0.6, check_obstacles=True)
                elif target_position == "right":
                    angle = int(20 * angle_multiplier)
                    speed = int(25 * speed_multiplier)
                    print(f"🎯 Target on right, gentle turn in place right then moving forward (angle: {angle}°, speed: {speed}%)")
                    turn_result = turn_in_place_right(angle, speed, 0.8)
                    print(f"Turn result: {turn_result}")
                    return move_forward(speed, 0.6, check_obstacles=True)
                else:  # center
                    speed = int(25 * speed_multiplier)
                    print(f"🎯 Target ahead, careful forward movement (speed: {speed}%)")
                    return move_forward(speed, 0.6, check_obstacles=True)
            
            else:  # Close to target
                if target_position == "left":
                    angle = int(10 * angle_multiplier)
                    speed = int(20 * speed_multiplier)
                    print(f"🎯 Target on left, fine turn in place left then moving forward (angle: {angle}°, speed: {speed}%)")
                    turn_result = turn_in_place_left(angle, speed, 0.5)
                    print(f"Turn result: {turn_result}")
                    return move_forward(speed, 0.4, check_obstacles=True)
                elif target_position == "right":
                    angle = int(10 * angle_multiplier)
                    speed = int(20 * speed_multiplier)
                    print(f"🎯 Target on right, fine turn in place right then moving forward (angle: {angle}°, speed: {speed}%)")
                    turn_result = turn_in_place_right(angle, speed, 0.5)
                    print(f"Turn result: {turn_result}")
                    return move_forward(speed, 0.4, check_obstacles=True)
                else:  # center
                    speed = int(20 * speed_multiplier)
                    print(f"🎯 Target ahead, fine forward movement (speed: {speed}%)")
                    return move_forward(speed, 0.4, check_obstacles=True)
                    
        except Exception as e:
            print(f"Error in movement strategy: {e}")
            # Fallback to simple forward movement
            return move_forward(25, 0.5, check_obstacles=True)
    
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
    
    def _quick_analyze_photo(self, photo_result: str, target_object: str, view_name: str) -> str:
        """Quickly analyze a single photo for target detection."""
        try:
            import re
            path_match = re.search(r': (.+\.jpg)', photo_result)
            if not path_match:
                return f"No photo available for {view_name} view"
            
            image_path = path_match.group(1)
            
            # Quick analysis prompt
            prompt_text = f"""Quick analysis of {view_name} view: Can you see a {target_object} in this image? 
            If yes, is it on the left, right, or center of this specific view? 
            If no, what objects do you see? 
            Respond with: TARGET: [visible/not visible] POSITION: [left/right/center] OBJECTS: [what you see]"""
            
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
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
                max_tokens=100
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Quick analysis error for {view_name}: {str(e)}"
    
    def _combine_panoramic_analysis(self, analysis_results: list, target_object: str) -> str:
        """Combine multiple view analyses into a comprehensive understanding."""
        try:
            # Find which views detected the target
            target_views = []
            all_objects = []
            
            for i, analysis in enumerate(analysis_results):
                view_names = ["left", "center", "right", "down", "up"]
                view_name = view_names[i] if i < len(view_names) else f"view_{i+1}"
                
                if "TARGET: visible" in analysis or "visible" in analysis.lower():
                    target_views.append(view_name)
                
                # Extract objects seen
                if "OBJECTS:" in analysis:
                    objects_part = analysis.split("OBJECTS:")[1].strip()
                    all_objects.append(f"{view_name}: {objects_part}")
            
            # Create combined analysis
            if target_views:
                if len(target_views) == 1:
                    position = target_views[0]
                    confidence = 8
                elif "center" in target_views:
                    position = "center"
                    confidence = 9
                else:
                    position = target_views[0]  # Use first detection
                    confidence = 7
                
                combined = f"""TARGET: visible
POSITION: {position} (detected in {', '.join(target_views)} views)
CONFIDENCE: {confidence}
DETAILS: Target {target_object} found in {len(target_views)} camera views: {', '.join(target_views)}. 
Spatial context: {'; '.join(all_objects[:3])}"""
            else:
                combined = f"""TARGET: not visible
POSITION: unknown
CONFIDENCE: 6
DETAILS: Target {target_object} not found in any camera view. 
Objects detected: {'; '.join(all_objects[:3])}"""
            
            return combined
            
        except Exception as e:
            return f"Panoramic analysis combination error: {str(e)}"

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
            
            # Analyze surroundings to identify the target object
            print(f"🎯 TARGET DESIGNATION: Looking for '{target_object}'")
            print(f"🔍 Analyzing surroundings to find {target_object}...")
            vision_analysis = self.analyze_surroundings(target_object, estimate_distance=True, multiple_photos=True)
            print(f"Vision Analysis: {vision_analysis}")
            
            # Extract and display what the robot actually sees as the target
            target_detected = self._extract_target_from_vision(vision_analysis)
            print(f"🎯 ROBOT'S TARGET: {target_detected}")
            
            # Wait a moment for system to stabilize
            print("⏳ Stabilizing sensors...")
            time.sleep(1.0)
            
            # Start navigation loop
            invalid_readings = 0
            max_invalid_readings = 8  # Increased - try ultrasonic more before camera
            consecutive_vision_attempts = 0
            max_vision_attempts = 2  # Reduced - use camera less often
            
            while self.navigation_active:
                # Get current distance with multiple attempts for stability
                current_distance = self._get_stable_distance()
                
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
                
                # Periodically re-analyze surroundings to ensure we're heading toward the right object
                if self.attempt_count % 3 == 0:  # Every 3 attempts (more frequent)
                    print("🔄 Re-analyzing surroundings...")
                    vision_analysis = self.analyze_surroundings(target_object, estimate_distance=True)
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
            return f"Navigating to {self.target_object} (target: {self.target_distance}cm). Current: {distance_result}"
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
