#!/usr/bin/env python3
"""
Debug Navigation Agent for PicarX Robot
Takes 5 pictures, sends to assistant, parses response - NO MOVEMENT
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

class DebugNavigationAgent:
    def __init__(self):
        """Initialize the Debug Navigation Agent."""
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
    def take_3_pictures(self, target_object: str = "nearest object") -> list:
        """Take 3 pictures from different camera angles (left, right, center)."""
        try:
            # Initialize camera
            camera_result = init_camera()
            print(f"📷 Camera: {camera_result}")
            
            # Define camera positions for 3-frame panorama
            camera_positions = [
                {"pan": -20, "tilt": 0, "name": "left"},
                {"pan": 20, "tilt": 0, "name": "right"}, 
                {"pan": 0, "tilt": 0, "name": "center"}
            ]
            
            image_paths = []
            print(f"📸 Taking 3-frame panorama for '{target_object}'...")
            
            for i, pos in enumerate(camera_positions):
                # Move camera to position
                pan_result = set_cam_pan_angle(pos["pan"])
                tilt_result = set_cam_tilt_angle(pos["tilt"])
                print(f"📷 Position {i+1}/3: {pos['name']} (pan={pos['pan']}°, tilt={pos['tilt']}°)")
                time.sleep(0.3)  # Wait for camera to settle
                
                # Take photo
                photo_result = capture_image(f"debug_{pos['name']}_{i+1}.jpg")
                print(f"📸 Photo {pos['name']}: {photo_result}")
                
                # Extract file path
                if "successfully" in photo_result.lower():
                    import re
                    path_match = re.search(r': (.+\.jpg)', photo_result)
                    if path_match:
                        image_paths.append(path_match.group(1))
                        print(f"✅ {pos['name']} image saved: {path_match.group(1)}")
                    else:
                        print(f"❌ Could not extract path for {pos['name']} photo")
                        return []
                else:
                    print(f"❌ Failed to capture {pos['name']} photo")
                    return []
            
            # Reset camera to center
            set_cam_pan_angle(0)
            set_cam_tilt_angle(0)
            print("📷 Camera reset to center position")
            
            return image_paths
            
        except Exception as e:
            print(f"❌ Error taking pictures: {str(e)}")
            return []
    
    def send_to_assistant(self, image_paths: list, target_object: str, max_retries: int = 3) -> str:
        """Send all 3 images to GPT-4 Vision assistant with retry logic."""
        for attempt in range(max_retries):
            try:
                print(f"🔄 Vision attempt {attempt + 1}/{max_retries}")
                
                if len(image_paths) != 3:
                    return f"❌ Expected 3 images, got {len(image_paths)}"
                
                # Prepare all images for the vision call
                view_names = ["left", "right", "center"]
                contents = []
                
                # Add instruction text
                contents.append({
                    "type": "text",
                    "text": f"You will receive 3 labeled views. Use them to pick the NEAREST {target_object} and tell me the best view."
                })
                
                # Add each image with label
                for i, (view, path) in enumerate(zip(view_names, image_paths)):
                    print(f"📤 Preparing {view} image: {path}")
                    
                    # Check if file exists and is readable
                    if not os.path.exists(path):
                        return f"❌ Image file not found: {path}"
                    
                    file_size = os.path.getsize(path)
                    if file_size == 0:
                        return f"❌ Image file is empty: {path}"
                    
                    print(f"📤 File size: {file_size} bytes")
                    
                    # Read and encode image
                    with open(path, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    
                    print(f"📤 Base64 length: {len(base64_image)}")
                    
                    # Add view label
                    contents.append({
                        "type": "text",
                        "text": f"VIEW: {view}"
                    })
                    
                    # Add image
                    contents.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
                
                print(f"📤 Sending {len(contents)} content parts to GPT-4 Vision...")
                print(f"📤 Image parts: {sum(1 for p in contents if p.get('type')=='image_url')}")
                
                # Create prompt
                prompt_text = f"""Analyze these 3 images taken from different camera angles (left, right, center) and find the nearest {target_object}.

STRICT OUTPUT FORMAT REQUIRED:

1. RANK the views by proximity to {target_object} (1=closest, 3=farthest):
RANKING: [view_name] [view_name] [view_name]

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
CENTER: [target visible? position? distance?]

Be CONSERVATIVE with distance estimates. If uncertain, estimate on the FAR side (safer)."""
                
                # Send to GPT-4 Vision
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt_text
                                }
                            ] + contents
                        }
                    ],
                    max_tokens=500
                )
                
                analysis = response.choices[0].message.content
                print(f"📥 Assistant response received ({len(analysis)} characters)")
                
                # Check if response looks valid (has our expected fields)
                if any(key in analysis for key in ["RANKING:", "BEST_VIEW:", "TARGET:", "POSITION:", "DISTANCE:", "CONFIDENCE:"]):
                    print(f"✅ Vision analysis successful on attempt {attempt + 1}")
                    return analysis
                else:
                    print(f"⚠️ Vision response incomplete on attempt {attempt + 1}")
                    print(f"Response: {analysis}")
                    if attempt < max_retries - 1:
                        print(f"🔄 Retrying in 1 second...")
                        time.sleep(1.0)
                        continue
                    else:
                        print(f"❌ All vision attempts failed")
                        return f"❌ Vision analysis failed after {max_retries} attempts. Last response: {analysis}"
                
            except Exception as e:
                print(f"⚠️ Vision attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"🔄 Retrying in 1 second...")
                    time.sleep(1.0)
                    continue
                else:
                    print(f"❌ All vision attempts failed")
                    return f"❌ Error sending to assistant after {max_retries} attempts: {str(e)}"
        
        return f"❌ Vision analysis failed after {max_retries} attempts"
    
    def parse_response(self, response: str) -> dict:
        """Parse the assistant's response into structured data."""
        try:
            import re
            
            parsed = {
                "ranking": None,
                "best_view": None,
                "target": None,
                "position": None,
                "distance": None,
                "confidence": None,
                "details": None,
                "view_analyses": {}
            }
            
            print("🔍 Parsing assistant response...")
            
            # Extract ranking
            ranking_match = re.search(r'RANKING:\s*([^\n]+)', response, re.IGNORECASE)
            if ranking_match:
                parsed["ranking"] = ranking_match.group(1).strip()
                print(f"📊 RANKING: {parsed['ranking']}")
            
            # Extract best view
            best_view_match = re.search(r'BEST_VIEW:\s*([^\n]+)', response, re.IGNORECASE)
            if best_view_match:
                parsed["best_view"] = best_view_match.group(1).strip()
                print(f"🎯 BEST_VIEW: {parsed['best_view']}")
            
            # Extract target
            target_match = re.search(r'TARGET:\s*([^\n]+)', response, re.IGNORECASE)
            if target_match:
                parsed["target"] = target_match.group(1).strip()
                print(f"👁️ TARGET: {parsed['target']}")
            
            # Extract position
            position_match = re.search(r'POSITION:\s*([^\n]+)', response, re.IGNORECASE)
            if position_match:
                parsed["position"] = position_match.group(1).strip()
                print(f"📍 POSITION: {parsed['position']}")
            
            # Extract distance
            distance_match = re.search(r'DISTANCE:\s*([^\n]+)', response, re.IGNORECASE)
            if distance_match:
                parsed["distance"] = distance_match.group(1).strip()
                print(f"📏 DISTANCE: {parsed['distance']}")
            
            # Extract confidence
            confidence_match = re.search(r'CONFIDENCE:\s*([^\n]+)', response, re.IGNORECASE)
            if confidence_match:
                parsed["confidence"] = confidence_match.group(1).strip()
                print(f"🎯 CONFIDENCE: {parsed['confidence']}")
            
            # Extract details
            details_match = re.search(r'DETAILS:\s*([^\n]+)', response, re.IGNORECASE)
            if details_match:
                parsed["details"] = details_match.group(1).strip()
                print(f"📝 DETAILS: {parsed['details']}")
            
            # Extract view analyses
            for view in ["LEFT", "RIGHT", "CENTER"]:
                view_match = re.search(f'{view}:\s*([^\n]+)', response, re.IGNORECASE)
                if view_match:
                    parsed["view_analyses"][view.lower()] = view_match.group(1).strip()
                    print(f"🔍 {view}: {parsed['view_analyses'][view.lower()]}")
            
            return parsed
            
        except Exception as e:
            print(f"❌ Error parsing response: {str(e)}")
            return {"error": str(e)}
    
    def turn_towards_object(self, parsed_data: dict) -> str:
        """Turn the car towards the nearest object based on parsed vision data."""
        try:
            best_view = parsed_data.get("best_view", "").lower()
            target = parsed_data.get("target", "").lower()
            confidence = parsed_data.get("confidence", "0")
            
            print(f"\n🔄 TURNING TOWARDS OBJECT")
            print(f"Best view: {best_view}")
            print(f"Target: {target}")
            print(f"Confidence: {confidence}")
            
            # Check if we have a valid target
            if target not in ["visible", "not visible"]:
                return f"❌ Invalid target status: {target}"
            
            if target == "not visible":
                return "❌ Target not visible, cannot turn towards it"
            
            # Check confidence (should be a number)
            try:
                conf_num = float(confidence)
                if conf_num < 5:
                    return f"⚠️ Low confidence ({conf_num}), not turning"
            except (ValueError, TypeError):
                return f"❌ Invalid confidence value: {confidence}"
            
            # Determine turn direction based on best view
            if best_view == "left":
                print("🔄 Turning left towards object...")
                turn_result = turn_in_place_left(30, 20, 1.0)  # 30 degrees, 20% speed, 1 second
                print(f"Initial turn: {turn_result}")
                
                # Now center the target using proportional control
                center_result = self.center_target_with_vision(parsed_data.get("target_object", "object"))
                return f"✅ Turned left: {turn_result}\n🎯 Centering: {center_result}"
                
            elif best_view == "right":
                print("🔄 Turning right towards object...")
                turn_result = turn_in_place_right(30, 20, 1.0)  # 30 degrees, 20% speed, 1 second
                print(f"Initial turn: {turn_result}")
                
                # Now center the target using proportional control
                center_result = self.center_target_with_vision(parsed_data.get("target_object", "object"))
                return f"✅ Turned right: {turn_result}\n🎯 Centering: {center_result}"
                
            elif best_view == "center":
                print("✅ Object already centered, no turning needed")
                return "✅ Object already centered"
                
            else:
                return f"❌ Unknown best view: {best_view}"
                
        except Exception as e:
            return f"❌ Error turning towards object: {str(e)}"
    
    def center_target_with_vision(self, target_object: str, max_attempts: int = 5, tolerance: float = 0.1) -> str:
        """Center the target using proportional control with vision feedback."""
        try:
            print(f"\n🎯 CENTERING TARGET WITH VISION")
            print(f"Target: {target_object}")
            print(f"Max attempts: {max_attempts}, Tolerance: {tolerance}")
            
            for attempt in range(max_attempts):
                print(f"\n🔄 Centering attempt {attempt + 1}/{max_attempts}")
                
                # Capture center frame
                print("📸 Capturing center frame...")
                photo_result = capture_image(f"center_attempt_{attempt + 1}.jpg")
                print(f"📸 Center photo: {photo_result}")
                
                # Extract file path
                import re
                path_match = re.search(r': (.+\.jpg)', photo_result)
                if not path_match:
                    return f"❌ Could not extract image path from: {photo_result}"
                
                image_path = path_match.group(1)
                
                # Send to assistant for HORIZ_OFFSET analysis with retry
                offset, confidence, centered = self.get_horizontal_offset_with_retry(image_path, target_object)
                
                if offset is None:
                    print(f"⚠️ Could not get horizontal offset on attempt {attempt + 1}")
                    if attempt < max_attempts - 1:
                        time.sleep(0.5)
                        continue
                    else:
                        return f"❌ Failed to get horizontal offset after {max_attempts} attempts"
                
                print(f"📊 HORIZ_OFFSET: {offset}, CONFIDENCE: {confidence}, CENTERED: {centered}")
                
                # Check if GPT clearly says the object is centered
                if centered and centered.lower() == "yes":
                    print(f"✅ Target clearly centered! GPT confirms object is at center of image.")
                    return f"✅ Target clearly centered after {attempt + 1} attempts. GPT confirmation: {centered}"
                
                # Fallback: Check if centered (within tolerance)
                if abs(offset) <= tolerance:
                    print(f"✅ Target centered! Offset: {offset} (tolerance: {tolerance})")
                    return f"✅ Target centered after {attempt + 1} attempts. Final offset: {offset:.3f}"
                
                # Calculate proportional turn angle
                # Scale factor: 1.0 means 1.0 offset = 10 degrees turn
                scale_factor = 10.0
                turn_angle = offset * scale_factor
                
                # Clamp turn angle to reasonable range
                turn_angle = max(-15, min(15, turn_angle))  # Clamp between -15 and +15 degrees
                
                print(f"🎯 Calculated turn: {turn_angle:.1f}° (from offset {offset:.3f})")
                
                # Perform the turn
                if turn_angle > 0:
                    print(f"🔄 Turning right by {turn_angle:.1f}°...")
                    turn_result = turn_in_place_right(abs(turn_angle), 15, 0.5)  # Small, precise turn
                elif turn_angle < 0:
                    print(f"🔄 Turning left by {abs(turn_angle):.1f}°...")
                    turn_result = turn_in_place_left(abs(turn_angle), 15, 0.5)  # Small, precise turn
                else:
                    print("✅ No turn needed (offset is 0)")
                    continue
                
                print(f"Turn result: {turn_result}")
                
                # Brief pause before next measurement
                time.sleep(0.3)
            
            return f"⚠️ Centering incomplete after {max_attempts} attempts. Final offset: {offset:.3f}"
            
        except Exception as e:
            return f"❌ Error centering target: {str(e)}"
    
    def get_horizontal_offset(self, image_path: str, target_object: str) -> tuple:
        """Get horizontal offset of target object from center using vision."""
        try:
            # Read and encode image
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Create prompt for horizontal offset analysis
            prompt_text = f"""Analyze this image and find the horizontal position of the {target_object}.

STRICT OUTPUT FORMAT REQUIRED:

HORIZ_OFFSET: [value between -1.0 and 1.0, where -1.0 = far left, 0.0 = center, 1.0 = far right]
CONFIDENCE: [1-10]
CENTERED: [yes/no - is the object clearly at the center of the image?]

The HORIZ_OFFSET should be:
- Negative if the object is to the left of center
- Positive if the object is to the right of center
- 0.0 if the object is perfectly centered
- Values closer to 0.0 mean more centered

CENTERED should be "yes" if the object is clearly at the center of the image, "no" otherwise."""
            
            # Send to GPT-4 Vision
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
            
            analysis = response.choices[0].message.content
            print(f"📥 Offset analysis: {analysis}")
            
            # Parse HORIZ_OFFSET, CONFIDENCE, and CENTERED
            import re
            
            offset_match = re.search(r'HORIZ_OFFSET:\s*([+-]?\d*\.?\d+)', analysis, re.IGNORECASE)
            confidence_match = re.search(r'CONFIDENCE:\s*(\d+)', analysis, re.IGNORECASE)
            centered_match = re.search(r'CENTERED:\s*(yes|no)', analysis, re.IGNORECASE)
            
            if offset_match and confidence_match and centered_match:
                offset = float(offset_match.group(1))
                confidence = int(confidence_match.group(1))
                centered = centered_match.group(1).lower()
                
                # Clamp offset to valid range
                offset = max(-1.0, min(1.0, offset))
                
                return offset, confidence, centered
            else:
                print(f"⚠️ Could not parse HORIZ_OFFSET, CONFIDENCE, or CENTERED from: {analysis}")
                return None, None, None
                
        except Exception as e:
            print(f"❌ Error getting horizontal offset: {str(e)}")
            return None, None, None
    
    def get_horizontal_offset_with_retry(self, image_path: str, target_object: str, max_retries: int = 3) -> tuple:
        """Get horizontal offset with retry logic - don't move if GPT doesn't respond properly."""
        for retry in range(max_retries):
            try:
                print(f"🔄 Vision retry {retry + 1}/{max_retries} for horizontal offset...")
                
                # Read and encode image
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
                # Create prompt for horizontal offset analysis
                prompt_text = f"""Analyze this image and find the horizontal position of the {target_object}.

STRICT OUTPUT FORMAT REQUIRED:

HORIZ_OFFSET: [value between -1.0 and 1.0, where -1.0 = far left, 0.0 = center, 1.0 = far right]
CONFIDENCE: [1-10]
CENTERED: [yes/no - is the object clearly at the center of the image?]

The HORIZ_OFFSET should be:
- Negative if the object is to the left of center
- Positive if the object is to the right of center
- 0.0 if the object is perfectly centered
- Values closer to 0.0 mean more centered

CENTERED should be "yes" if the object is clearly at the center of the image, "no" otherwise."""
                
                # Send to GPT-4 Vision
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
                
                analysis = response.choices[0].message.content
                print(f"📥 Offset analysis (retry {retry + 1}): {analysis}")
                
                # Parse HORIZ_OFFSET, CONFIDENCE, and CENTERED
                import re
                
                offset_match = re.search(r'HORIZ_OFFSET:\s*([+-]?\d*\.?\d+)', analysis, re.IGNORECASE)
                confidence_match = re.search(r'CONFIDENCE:\s*(\d+)', analysis, re.IGNORECASE)
                centered_match = re.search(r'CENTERED:\s*(yes|no)', analysis, re.IGNORECASE)
                
                if offset_match and confidence_match and centered_match:
                    offset = float(offset_match.group(1))
                    confidence = int(confidence_match.group(1))
                    centered = centered_match.group(1).lower()
                    
                    # Clamp offset to valid range
                    offset = max(-1.0, min(1.0, offset))
                    
                    print(f"✅ Valid response on retry {retry + 1}: offset={offset}, confidence={confidence}, centered={centered}")
                    return offset, confidence, centered
                else:
                    print(f"⚠️ Incomplete response on retry {retry + 1}: {analysis}")
                    if retry < max_retries - 1:
                        print(f"🔄 Retrying in 1 second...")
                        time.sleep(1.0)
                        continue
                    else:
                        print(f"❌ All retries failed for horizontal offset")
                        return None, None, None
                
            except Exception as e:
                print(f"⚠️ Vision retry {retry + 1} failed: {str(e)}")
                if retry < max_retries - 1:
                    print(f"🔄 Retrying in 1 second...")
                    time.sleep(1.0)
                    continue
                else:
                    print(f"❌ All retries failed for horizontal offset")
                    return None, None, None
        
        return None, None, None
    
    def approach_object_slowly(self, target_object: str, target_distance: float, max_steps: int = 20) -> str:
        """Approach object slowly using both ultrasonic and vision distance measurements."""
        try:
            print(f"\n🚶 APPROACHING OBJECT SLOWLY")
            print(f"Target: {target_object}")
            print(f"Target distance: {target_distance}cm")
            print(f"Max steps: {max_steps}")
            
            for step in range(max_steps):
                print(f"\n🔄 Approach step {step + 1}/{max_steps}")
                
                # Get ultrasonic distance
                print("📡 Getting ultrasonic distance...")
                ultrasonic_result = get_ultrasonic_distance()
                print(f"📡 Ultrasonic: {ultrasonic_result}")
                
                # Parse ultrasonic distance
                ultrasonic_distance = self.parse_ultrasonic_distance(ultrasonic_result)
                
                # Get vision distance
                print("📸 Getting vision distance...")
                photo_result = capture_image(f"approach_step_{step + 1}.jpg")
                print(f"📸 Approach photo: {photo_result}")
                
                # Extract file path
                import re
                path_match = re.search(r': (.+\.jpg)', photo_result)
                if not path_match:
                    print("⚠️ Could not extract image path, using ultrasonic only")
                    vision_distance = None
                else:
                    image_path = path_match.group(1)
                    vision_distance = self.get_vision_distance(image_path, target_object)
                    print(f"📷 Vision distance: {vision_distance}cm" if vision_distance else "📷 Vision distance: failed")
                
                # Determine current distance (prefer ultrasonic, fallback to vision)
                if ultrasonic_distance is not None and ultrasonic_distance > 0:
                    current_distance = ultrasonic_distance
                    distance_source = "ultrasonic"
                elif vision_distance is not None and vision_distance > 0:
                    current_distance = vision_distance
                    distance_source = "vision"
                else:
                    print("⚠️ No valid distance measurement, stopping for safety")
                    stop()
                    return f"❌ No valid distance measurement on step {step + 1}, stopped for safety"
                
                print(f"📏 Current distance: {current_distance:.1f}cm ({distance_source})")
                
                # Check if we've reached the target distance
                if current_distance <= target_distance:
                    stop()
                    print(f"✅ Reached target distance! Final distance: {current_distance:.1f}cm")
                    return f"✅ Successfully approached to {current_distance:.1f}cm (target: {target_distance}cm) after {step + 1} steps"
                
                # Safety check - too close
                if current_distance < 5.0:
                    stop()
                    print(f"⚠️ Safety stop! Too close at {current_distance:.1f}cm")
                    return f"⚠️ Safety stop at {current_distance:.1f}cm (too close)"
                
                # Calculate movement distance (small steps)
                movement_distance = min(5.0, current_distance - target_distance)  # Move up to 5cm closer
                if movement_distance < 1.0:
                    movement_distance = 1.0  # Minimum 1cm step
                
                print(f"🚶 Moving forward {movement_distance:.1f}cm...")
                
                # Move forward slowly
                move_result = move_forward(movement_distance, 15)  # 15% speed for slow movement
                print(f"🚶 Move result: {move_result}")
                
                # Brief pause between steps
                time.sleep(0.5)
            
            # If we get here, we've reached max steps
            stop()
            return f"⚠️ Reached maximum steps ({max_steps}) without reaching target distance"
            
        except Exception as e:
            stop()
            return f"❌ Error approaching object: {str(e)}"
    
    def parse_ultrasonic_distance(self, ultrasonic_result: str) -> float:
        """Parse distance from ultrasonic sensor result."""
        try:
            import re
            # Look for pattern like "Ultrasound detected an obstacle at 25.3 cm"
            match = re.search(r'at ([\d.]+)\s*cm', ultrasonic_result)
            if match:
                distance = float(match.group(1))
                # Validate distance (reasonable range)
                if 2.0 <= distance <= 400.0:
                    return distance
            return None
        except Exception as e:
            print(f"⚠️ Error parsing ultrasonic distance: {str(e)}")
            return None
    
    def get_vision_distance(self, image_path: str, target_object: str) -> float:
        """Get distance estimate from vision analysis."""
        try:
            # Read and encode image
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Create prompt for distance analysis
            prompt_text = f"""Analyze this image and estimate the distance to the {target_object}.

STRICT OUTPUT FORMAT REQUIRED:

DISTANCE: [specific cm estimate]
CONFIDENCE: [1-10]

Be CONSERVATIVE with distance estimates. If uncertain, estimate on the FAR side (safer)."""
            
            # Send to GPT-4 Vision
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
            
            analysis = response.choices[0].message.content
            print(f"📥 Distance analysis: {analysis}")
            
            # Parse DISTANCE
            import re
            distance_match = re.search(r'DISTANCE:\s*([\d.]+)', analysis, re.IGNORECASE)
            
            if distance_match:
                distance = float(distance_match.group(1))
                # Validate distance (reasonable range)
                if 5.0 <= distance <= 500.0:
                    return distance
            return None
                
        except Exception as e:
            print(f"❌ Error getting vision distance: {str(e)}")
            return None
    
    def debug_navigation(self, target_object: str = "nearest object", turn_towards: bool = True, approach_distance: float = None) -> dict:
        """Complete debug workflow: take pictures, send to assistant, parse response, optionally turn, optionally approach."""
        print("🔍 DEBUG NAVIGATION WORKFLOW")
        print("=" * 50)
        
        # Step 1: Take 3 pictures
        print("\n📸 STEP 1: Taking 3 pictures...")
        image_paths = self.take_3_pictures(target_object)
        
        if not image_paths:
            return {"error": "Failed to take pictures"}
        
        print(f"✅ Successfully captured {len(image_paths)} images")
        
        # Step 2: Send to assistant
        print("\n📤 STEP 2: Sending to GPT-4 Vision...")
        response = self.send_to_assistant(image_paths, target_object)
        
        if response.startswith("❌"):
            return {"error": response}
        
        print("✅ Successfully sent to assistant")
        
        # Step 2.5: Print raw response
        print("\n📄 RAW ASSISTANT RESPONSE:")
        print("=" * 50)
        print(response)
        print("=" * 50)
        
        # Step 3: Parse response
        print("\n🔍 STEP 3: Parsing response...")
        parsed = self.parse_response(response)
        
        # Step 4: Turn towards object (if requested and successful)
        turn_result = None
        if turn_towards and "error" not in parsed:
            print("\n🔄 STEP 4: Turning towards object...")
            turn_result = self.turn_towards_object(parsed)
            print(f"Turn result: {turn_result}")
        elif turn_towards:
            print("\n⚠️ STEP 4: Skipping turn due to parsing error")
        else:
            print("\n⏭️ STEP 4: Skipping turn (turn_towards=False)")
        
        # Step 5: Approach object (if distance specified)
        approach_result = None
        if approach_distance is not None and "error" not in parsed:
            print(f"\n🚶 STEP 5: Approaching object to {approach_distance}cm...")
            approach_result = self.approach_object_slowly(target_object, approach_distance)
            print(f"Approach result: {approach_result}")
        elif approach_distance is not None:
            print("\n⚠️ STEP 5: Skipping approach due to parsing error")
        else:
            print("\n⏭️ STEP 5: Skipping approach (no distance specified)")
        
        print("\n📊 FINAL RESULTS:")
        print("=" * 30)
        for key, value in parsed.items():
            if key != "view_analyses":
                print(f"{key.upper()}: {value}")
        
        print("\nVIEW ANALYSES:")
        for view, analysis in parsed.get("view_analyses", {}).items():
            print(f"  {view.upper()}: {analysis}")
        
        if turn_result:
            print(f"\nTURN RESULT: {turn_result}")
        
        if approach_result:
            print(f"\nAPPROACH RESULT: {approach_result}")
        
        return {
            "image_paths": image_paths,
            "raw_response": response,
            "parsed_data": parsed,
            "turn_result": turn_result,
            "approach_result": approach_result
        }

def main():
    """Interactive debug navigation agent."""
    print("🔍 PicarX Debug Navigation Agent")
    print("=" * 40)
    print("This agent will:")
    print("1. Take 3 pictures from different angles (left, right, center)")
    print("2. Send them to GPT-4 Vision")
    print("3. Parse the response")
    print("4. Turn towards the nearest object (if successful)")
    print("5. Center the target using vision feedback")
    print("6. Optionally approach the object to a specified distance")
    print("7. Show you the results")
    print()
    
    agent = DebugNavigationAgent()
    
    while True:
        try:
            command = input("DebugNav> ").strip()
            
            if command.lower() in ["quit", "exit"]:
                print("👋 Goodbye!")
                break
            elif command.lower() == "help":
                print("Commands:")
                print("- 'debug' - Run full debug workflow with turning")
                print("- 'debug nearest object' - Debug with specific target")
                print("- 'debug no-turn' - Debug without turning")
                print("- 'debug nearest object no-turn' - Debug with target, no turning")
                print("- 'debug approach 30' - Debug with approach to 30cm")
                print("- 'debug nearest object approach 20' - Debug with target and approach to 20cm")
                print("- 'quit' - Exit")
            elif command.lower() == "debug":
                result = agent.debug_navigation(turn_towards=True)
                if "error" in result:
                    print(f"❌ Error: {result['error']}")
                else:
                    print("✅ Debug workflow completed successfully!")
            elif command.lower() == "debug no-turn":
                result = agent.debug_navigation(turn_towards=False)
                if "error" in result:
                    print(f"❌ Error: {result['error']}")
                else:
                    print("✅ Debug workflow completed successfully!")
            elif command.lower().startswith("debug "):
                # Extract everything after "debug "
                remaining = command[6:].strip()
                
                # Parse command with approach distance
                parts = remaining.split()
                target = "nearest object"
                turn_towards = True
                approach_distance = None
                
                # Look for approach distance
                if "approach" in parts:
                    approach_idx = parts.index("approach")
                    if approach_idx + 1 < len(parts):
                        try:
                            approach_distance = float(parts[approach_idx + 1])
                            # Remove approach parts from target
                            parts = parts[:approach_idx]
                        except ValueError:
                            print("❌ Invalid approach distance. Use format: 'debug target approach 30'")
                            continue
                
                # Determine target and turn_towards
                if not parts or parts[0] == "no-turn":
                    target = "nearest object"
                    turn_towards = parts[0] != "no-turn" if parts else True
                elif parts[-1] == "no-turn":
                    target = " ".join(parts[:-1])
                    turn_towards = False
                else:
                    target = " ".join(parts)
                    turn_towards = True
                
                # Run debug navigation
                result = agent.debug_navigation(target, turn_towards, approach_distance)
                
                if "error" in result:
                    print(f"❌ Error: {result['error']}")
                else:
                    print("✅ Debug workflow completed successfully!")
            else:
                print("❌ Unknown command. Type 'help' for commands or 'quit' to exit.")
                
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
