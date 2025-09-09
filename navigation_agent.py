#!/usr/bin/env python3

import sys
import os
import time
from typing import Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from primitives import *
from openai import OpenAI
from keys import OPENAI_API_KEY
import base64

class NavigationAgent:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.target_distance = None
        self.target_object = None
        self.navigation_active = False
        self.attempt_count = 0
        
        self.locked_target = None
        self.lock_confidence = 0.0
        self.lock_cycles = 0
        self.max_lock_cycles = 5
        self.consecutive_consistent_frames = 0
        self.last_position = None
        self.keys_lock = False
        
    def analyze_surroundings(self, target_object: str, estimate_distance: bool = False, panoramic: bool = False) -> str:
        try:
            camera_result = init_camera()
            print(f"Camera: {camera_result}")
            
            if panoramic:
                print("📸 Taking 3-frame panorama for comprehensive analysis...")
                camera_positions = [
                    {"pan": -20, "tilt": 0, "name": "left"},
                    {"pan": 20, "tilt": 0, "name": "right"}, 
                    {"pan": 0, "tilt": 0, "name": "center"}
                ]
                
                photo_results = []
                for pos in camera_positions:
                    set_cam_pan_angle(pos["pan"])
                    set_cam_tilt_angle(pos["tilt"])
                    time.sleep(0.5)
                    
                    photo_result = capture_image(f"panorama_{pos['name']}.jpg")
                    photo_results.append(photo_result)
                    print(f"📸 {pos['name']}: {photo_result}")
                
                set_cam_pan_angle(0)
                set_cam_tilt_angle(0)
                
                return self._analyze_panorama_images(photo_results, target_object, estimate_distance)
            else:
                photo_result = capture_image("single_view.jpg")
                print(f"📸 Single view: {photo_result}")
                return self._analyze_single_image(photo_result, target_object, estimate_distance)
                
        except Exception as e:
            return f"Surroundings analysis failed: {str(e)}"

    def _analyze_panorama_images(self, photo_results: list, target_object: str, estimate_distance: bool = False) -> str:
        try:
            image_paths = []
            for result in photo_results:
                import re
                path_match = re.search(r': (.+\.jpg)', result)
                if path_match:
                    image_paths.append(path_match.group(1))
            
            if not image_paths:
                return "❌ No valid images captured for panorama analysis"
            
            print(f"🔍 Analyzing {len(image_paths)} panorama images for '{target_object}'...")
            
            for attempt in range(3):
                try:
                    analysis = self._send_panorama_to_assistant(image_paths, target_object, estimate_distance)
                    
                    if self._is_valid_panorama_response(analysis):
                        print(f"✅ Valid panorama analysis received on attempt {attempt + 1}")
                        return analysis
                    else:
                        print(f"⚠️ Invalid panorama response on attempt {attempt + 1}, retrying...")
                        time.sleep(1)
                        
                except Exception as e:
                    print(f"⚠️ Panorama analysis attempt {attempt + 1} failed: {str(e)}")
                    if attempt < 2:
                        time.sleep(1)
                    else:
                        return f"❌ Panorama analysis failed after 3 attempts: {str(e)}"
            
            return "❌ Panorama analysis failed after 3 attempts"
            
        except Exception as e:
            return f"Panorama analysis error: {str(e)}"

    def _is_valid_panorama_response(self, response: str) -> bool:
        required_fields = ["RANKING:", "BEST_VIEW:", "TARGET:", "POSITION:", "DISTANCE:", "CONFIDENCE:"]
        
        has_all_fields = all(field in response for field in required_fields)
        
        invalid_patterns = [
            "I'm unable to display the specific analysis",
            "however I can guide you through",
            "I'm unable to provide specific distance estimates",
            "but I can offer an analysis based on visible cues",
            "unable to"
        ]
        
        has_invalid_pattern = any(pattern in response for pattern in invalid_patterns)
        
        return has_all_fields and not has_invalid_pattern

    def _send_panorama_to_assistant(self, image_paths: list, target_object: str, estimate_distance: bool = False) -> str:
        try:
            images = []
            for i, path in enumerate(image_paths):
                with open(path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    images.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "low"
                        }
                    })
            
            distance_instruction = ""
            if estimate_distance:
                distance_instruction = " Also estimate the distance to the nearest object in centimeters."
            
            prompt = f"""Analyze these 3 images (left, right, center) to find the nearest '{target_object}'. 

For each image, provide:
- RANKING: 1-3 (1=best view of target)
- BEST_VIEW: left/right/center
- TARGET: Description of what you see
- POSITION: left/center/right, foreground/background, size estimate
- DISTANCE: Estimated distance in cm{distance_instruction}
- CONFIDENCE: 1-10

Format your response exactly as:
RANKING: 1
BEST_VIEW: center
TARGET: [description]
POSITION: center, foreground, small
DISTANCE: 25 cm
CONFIDENCE: 8

RANKING: 2
BEST_VIEW: left
TARGET: [description]
POSITION: left, background, medium
DISTANCE: 45 cm
CONFIDENCE: 6

RANKING: 3
BEST_VIEW: right
TARGET: [description]
POSITION: right, foreground, large
DISTANCE: 15 cm
CONFIDENCE: 9"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            *images
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Vision analysis error: {str(e)}"

    def _analyze_single_image(self, photo_result: str, target_object: str, estimate_distance: bool = False) -> str:
        try:
            import re
            path_match = re.search(r': (.+\.jpg)', photo_result)
            if not path_match:
                return "❌ Could not extract image path from photo result"
            
            image_path = path_match.group(1)
            
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            distance_instruction = ""
            if estimate_distance:
                distance_instruction = " Also estimate the distance to the nearest object in centimeters."
            
            prompt = f"""Analyze this image to find the nearest '{target_object}'. 

Provide:
- TARGET: Description of what you see
- POSITION: left/center/right, foreground/background, size estimate
- DISTANCE: Estimated distance in cm{distance_instruction}
- CONFIDENCE: 1-10

Format your response exactly as:
TARGET: [description]
POSITION: center, foreground, small
DISTANCE: 25 cm
CONFIDENCE: 8"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Single image analysis error: {str(e)}"

    def _extract_best_view_and_direction(self, vision_analysis: str) -> tuple:
        try:
            lines = vision_analysis.split('\n')
            best_view = "center"
            target_direction = "center"
            
            for line in lines:
                if line.startswith("BEST_VIEW:"):
                    best_view = line.split(":", 1)[1].strip().lower()
                elif line.startswith("POSITION:"):
                    position = line.split(":", 1)[1].strip().lower()
                    if "left" in position:
                        target_direction = "left"
                    elif "right" in position:
                        target_direction = "right"
                    else:
                        target_direction = "center"
                    break
            
            return best_view, target_direction
            
        except Exception as e:
            print(f"Error extracting view info: {str(e)}")
            return "center", "center"

    def _extract_target_from_vision(self, vision_analysis: str) -> str:
        try:
            lines = vision_analysis.split('\n')
            for line in lines:
                if line.startswith("TARGET:"):
                    return line.split(":", 1)[1].strip()
            return "unknown object"
        except:
            return "unknown object"

    def get_centering_instructions_with_retry(self, image_path: str, target_object: str, max_retries: int = 5) -> dict:
        for attempt in range(max_retries):
            try:
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
                prompt = f"""Analyze this image to center the '{target_object}'. The target is already identified as '{target_object}'.

Provide:
- TARGET_VISIBLE: true/false
- POSITION: left/center/right
- HORIZ_OFFSET: -100 to +100 (negative=left, positive=right, 0=centered)
- TURN_DIRECTION: left/right/none
- TURN_ANGLE: degrees to turn (0-45)
- TURN_DURATION: seconds (Speed 50, 2.1 seconds = 90 degrees)
- CENTERED: true/false
- CONFIDENCE: 1-10

Format exactly as:
TARGET_VISIBLE: true
POSITION: right
HORIZ_OFFSET: 30
TURN_DIRECTION: right
TURN_ANGLE: 15
TURN_DURATION: 0.7
CENTERED: false
CONFIDENCE: 8"""

                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}",
                                        "detail": "low"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=300
                )
                
                analysis = response.choices[0].message.content
                instructions = self._parse_centering_instructions(analysis)
                
                if instructions:
                    return instructions
                else:
                    print(f"⚠️ Invalid centering response on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        time.sleep(0.5)
                    
            except Exception as e:
                print(f"⚠️ Centering attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(0.5)
        
        return None

    def _parse_centering_instructions(self, analysis: str) -> dict:
        try:
            lines = analysis.split('\n')
            instructions = {}
            
            for line in lines:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().upper()
                    value = value.strip()
                    
                    if key == "TARGET_VISIBLE":
                        instructions["target_visible"] = value.lower() == "true"
                    elif key == "POSITION":
                        instructions["position"] = value.lower()
                    elif key == "HORIZ_OFFSET":
                        try:
                            instructions["horiz_offset"] = int(value)
                        except:
                            instructions["horiz_offset"] = 0
                    elif key == "TURN_DIRECTION":
                        instructions["turn_direction"] = value.lower()
                    elif key == "TURN_ANGLE":
                        try:
                            instructions["turn_angle"] = float(value)
                        except:
                            instructions["turn_angle"] = 0
                    elif key == "TURN_DURATION":
                        try:
                            instructions["turn_duration"] = float(value)
                        except:
                            instructions["turn_duration"] = 0.5
                    elif key == "CENTERED":
                        instructions["centered"] = value.lower() == "true"
                    elif key == "CONFIDENCE":
                        try:
                            instructions["confidence"] = int(value)
                        except:
                            instructions["confidence"] = 5
            
            return instructions
            
        except Exception as e:
            print(f"Error parsing centering instructions: {str(e)}")
            return None

    def center_target_until_centered(self, target_object: str) -> str:
        try:
            print(f"\n🎯 CENTERING TARGET (UNTIL CENTERED)")
            print(f"Target: {target_object}")
            print("🔄 Will continue centering until GPT confirms target is centered...")

            attempt = 0
            max_attempts = 50
            
            while attempt < max_attempts:
                attempt += 1
                print(f"\n🔄 Centering attempt {attempt}")

                print("📸 Capturing center frame...")
                photo_result = capture_image(f"center_attempt_{attempt}.jpg")
                print(f"📸 Center photo: {photo_result}")

                import re
                path_match = re.search(r': (.+\.jpg)', photo_result)
                if not path_match:
                    print(f"❌ Could not extract image path from: {photo_result}")
                    if attempt < max_attempts - 1:
                        time.sleep(0.5)
                        continue
                    else:
                        return f"❌ Could not extract image path after {attempt} attempts"

                image_path = path_match.group(1)

                print(f"🎯 Getting centering instructions...")
                turn_instructions = self.get_centering_instructions_with_retry(image_path, target_object, max_retries=5)

                if turn_instructions is None:
                    print(f"⚠️ Could not get valid instructions on attempt {attempt}")
                    if attempt < max_attempts - 1:
                        time.sleep(0.5)
                        continue
                    else:
                        return f"❌ Failed to get valid instructions after {attempt} attempts"

                print(f"📊 TURN INSTRUCTIONS: {turn_instructions}")

                if turn_instructions.get("centered", False):
                    print(f"✅ Target centered! GPT confirmation: {turn_instructions.get('centered')}")
                    return f"✅ Target centered after {attempt} attempts. GPT confirmation: {turn_instructions.get('centered')}"

                if not turn_instructions.get("target_visible", False):
                    print(f"⚠️ Target not visible on attempt {attempt}")
                    if attempt < max_attempts - 1:
                        time.sleep(0.5)
                        continue
                    else:
                        return f"❌ Target not visible after {attempt} attempts"

                turn_direction = turn_instructions.get("turn_direction", "none")
                turn_angle = turn_instructions.get("turn_angle", 0)
                turn_duration = turn_instructions.get("turn_duration", 0.5)
                confidence = turn_instructions.get("confidence", 5)

                print(f"📊 Position: {turn_instructions.get('position')}")
                print(f"📊 Turn direction: {turn_direction}")
                print(f"📊 Turn angle: {turn_angle}°")
                print(f"📊 Turn duration: {turn_duration}s")
                print(f"📊 Confidence: {confidence}/10")

                if confidence < 5:
                    print(f"⚠️ Low confidence ({confidence}/10) on attempt {attempt}")
                    if attempt < max_attempts - 1:
                        time.sleep(0.5)
                        continue
                    else:
                        return f"⚠️ Low confidence after {attempt} attempts"

                position = turn_instructions.get("position", "center")
                logic_error = False
                
                if turn_direction == "left" and turn_angle > 0:
                    if position == "right":
                        logic_error = True
                        print(f"⚠️ Logic error: GPT says position=right but turn_direction=left. This seems backwards.")
                elif turn_direction == "right" and turn_angle > 0:
                    if position == "left":
                        logic_error = True
                        print(f"⚠️ Logic error: GPT says position=left but turn_direction=right. This seems backwards.")
                
                if logic_error:
                    print(f"🔄 Taking fresh picture to double-check target position...")
                    fresh_photo_result = capture_image(f"logic_check_attempt_{attempt}.jpg")
                    print(f"📸 Fresh photo: {fresh_photo_result}")
                    
                    fresh_path_match = re.search(r': (.+\.jpg)', fresh_photo_result)
                    if fresh_path_match:
                        fresh_image_path = fresh_path_match.group(1)
                        
                        print(f"🎯 Getting fresh centering instructions...")
                        fresh_instructions = self.get_centering_instructions_with_retry(fresh_image_path, target_object, max_retries=3)
                        
                        if fresh_instructions and fresh_instructions.get("target_visible", False):
                            print(f"📊 FRESH INSTRUCTIONS: {fresh_instructions}")
                            fresh_position = fresh_instructions.get("position", "center")
                            fresh_direction = fresh_instructions.get("turn_direction", "none")
                            fresh_angle = fresh_instructions.get("turn_angle", 0)
                            fresh_duration = fresh_instructions.get("turn_duration", 0.5)
                            
                            if (fresh_position == "right" and fresh_direction == "right") or \
                               (fresh_position == "left" and fresh_direction == "left") or \
                               fresh_direction == "none":
                                print(f"✅ Fresh instructions look correct, using them...")
                                turn_direction = fresh_direction
                                turn_angle = fresh_angle
                                turn_duration = fresh_duration
                                position = fresh_position
                                logic_error = False
                            else:
                                print(f"⚠️ Fresh instructions also have logic issues, proceeding with correction...")
                        else:
                            print(f"⚠️ Could not get fresh instructions, proceeding with correction...")
                    else:
                        print(f"⚠️ Could not extract fresh image path, proceeding with correction...")
                
                if turn_direction == "left" and turn_angle > 0:
                    if logic_error and position == "right":
                        print(f"🔄 Correcting: Turning RIGHT to center target on the right...")
                        turn_result = turn_in_place_right(turn_angle, 50, turn_duration)
                    else:
                        print(f"🔄 Turning left by {turn_angle}° for {turn_duration}s...")
                        turn_result = turn_in_place_left(turn_angle, 50, turn_duration)
                elif turn_direction == "right" and turn_angle > 0:
                    if logic_error and position == "left":
                        print(f"🔄 Correcting: Turning LEFT to center target on the left...")
                        turn_result = turn_in_place_left(turn_angle, 50, turn_duration)
                    else:
                        print(f"🔄 Turning right by {turn_angle}° for {turn_duration}s...")
                        turn_result = turn_in_place_right(turn_angle, 50, turn_duration)
                elif turn_direction == "none" or turn_angle == 0:
                    print("✅ No turn needed")
                    return f"✅ No turn needed after {attempt} attempts (GPT says none required)"
                else:
                    print(f"❌ Invalid turn instructions: direction={turn_direction}, angle={turn_angle}")
                    if attempt < max_attempts - 1:
                        time.sleep(0.5)
                        continue
                    else:
                        return f"❌ Invalid turn instructions after {attempt} attempts"
                
                print(f"Turn result: {turn_result}")

                time.sleep(0.3)

            return f"⚠️ Centering incomplete after {max_attempts} attempts (safety limit reached)"

        except Exception as e:
            return f"❌ Error centering target: {str(e)}"

    def approach_object_slowly(self, target_object: str, stop_distance: float) -> str:
        try:
            print(f"\n🚀 APPROACHING OBJECT SLOWLY")
            print(f"Target: {target_object}")
            print(f"Stop distance: {stop_distance}cm")
            
            step_count = 0
            max_steps = 20
            
            while step_count < max_steps:
                step_count += 1
                print(f"\n🚀 Step {step_count}/{max_steps}")
                
                print("📸 Taking picture for distance estimation...")
                photo_result = capture_image(f"approach_step_{step_count}.jpg")
                print(f"📸 Photo: {photo_result}")
                
                import re
                path_match = re.search(r': (.+\.jpg)', photo_result)
                if not path_match:
                    print(f"❌ Could not extract image path")
                    return f"❌ Could not extract image path on step {step_count}"
                
                image_path = path_match.group(1)
                
                ultrasonic_distance = self.parse_ultrasonic_distance(get_ultrasonic_distance())
                vision_distance = self.get_vision_distance(image_path, target_object)
                
                print(f"📊 Ultrasonic: {ultrasonic_distance}cm")
                print(f"📊 Vision: {vision_distance}cm")
                
                if ultrasonic_distance is not None and 2.0 <= ultrasonic_distance <= 400.0:
                    current_distance = ultrasonic_distance
                    distance_source = "ultrasonic"
                elif vision_distance is not None and 2.0 <= vision_distance <= 500.0:
                    current_distance = vision_distance
                    distance_source = "vision"
                else:
                    current_distance = 10.0
                    distance_source = "fallback"
                    print(f"⚠️ Both sensors unreliable, using conservative fallback: {current_distance}cm")
                
                print(f"📊 Using {distance_source} distance: {current_distance}cm")
                
                if current_distance <= stop_distance:
                    print(f"✅ Reached target distance! Current: {current_distance}cm, Target: {stop_distance}cm")
                    return f"✅ Successfully approached {target_object} to {current_distance}cm (target: {stop_distance}cm)"
                
                remaining_distance = current_distance - stop_distance
                step_size = max(1.0, min(8.0, remaining_distance * 0.3))
                
                print(f"📊 Remaining: {remaining_distance:.1f}cm, Step size: {step_size:.1f}cm")
                
                if step_size < 1.0:
                    print(f"✅ Very close to target, stopping")
                    return f"✅ Successfully approached {target_object} to {current_distance}cm (very close to target)"
                
                step_duration = step_size / 20.0
                step_duration = max(0.1, min(step_duration, 1.0))
                
                print(f"🚀 Moving forward {step_size:.1f}cm for {step_duration:.2f}s...")
                move_result = move_forward(30, step_duration, check_obstacles=False)
                print(f"🚀 Move result: {move_result}")
                
                time.sleep(0.2)
            
            return f"⚠️ Approach incomplete after {max_steps} steps (safety limit reached)"

        except Exception as e:
            return f"❌ Error approaching object: {str(e)}"

    def get_vision_distance(self, image_path: str, target_object: str) -> float:
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            prompt = f"""Analyze this image to estimate the distance to the '{target_object}'.

Provide only:
DISTANCE: [number] cm

Be specific and accurate. Look for size cues, perspective, and depth indicators."""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=100
            )
            
            analysis = response.choices[0].message.content
            return self._extract_distance_from_vision(analysis)
            
        except Exception as e:
            print(f"Vision distance error: {str(e)}")
            return None

    def _extract_distance_from_vision(self, vision_analysis: str) -> float:
        try:
            import re
            
            distance_line = re.search(r'DISTANCE:\s*([^\n]+)', vision_analysis, re.IGNORECASE)
            text = distance_line.group(1) if distance_line else vision_analysis
            
            patterns = [
                r'(\d+(?:\.\d+)?)\s*cm',
                r'about\s+(\d+(?:\.\d+)?)\s*cm',
                r'approximately\s+(\d+(?:\.\d+)?)\s*cm',
                r'(\d+(?:\.\d+)?)\s*centimeters',
                r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*cm',
                r'(\d+(?:\.\d+)?)\s*to\s*(\d+(?:\.\d+)?)\s*cm',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    if isinstance(matches[0], tuple):
                        distance = float(matches[0][0])
                    else:
                        distance = float(matches[0])
                    
                    if 2.0 <= distance <= 500.0:
                        return distance
                    else:
                        print(f"⚠️ Vision distance {distance}cm outside reasonable range (2-500cm)")
                        return None
            
            return None
            
        except Exception as e:
            print(f"Error extracting distance from vision: {str(e)}")
            return None

    def parse_ultrasonic_distance(self, ultrasonic_result: str) -> float:
        try:
            import re
            match = re.search(r'([-+]?\d+(?:\.\d+)?)\s*cm', ultrasonic_result)
            if match:
                distance = float(match.group(1))
                if 2.0 <= distance <= 400.0:
                    return distance
            return None
        except:
            return None

    def navigate_to_distance(self, target_distance: float, target_object: str = "object") -> str:
        try:
            self.target_distance = target_distance
            self.target_object = target_object
            self.navigation_active = True
            
            reset_result = reset()
            print(f"Reset: {reset_result}")
            
            self.locked_target = None
            self.lock_confidence = 0.0
            self.lock_cycles = 0
            self.consecutive_consistent_frames = 0
            self.last_position = None
            self.keys_lock = False
            print("🔓 Target lock reset for new navigation task")
            
            print(f"🎯 TARGET DESIGNATION: Looking for '{target_object}'")
            print(f"🔍 Taking 3-frame panorama to find {target_object}...")
            vision_analysis = self.analyze_surroundings(target_object, estimate_distance=True, panoramic=True)
            print(f"Panorama Analysis: {vision_analysis}")
            
            best_view, target_direction = self._extract_best_view_and_direction(vision_analysis)
            print(f"🎯 BEST VIEW: {best_view} -> Direction: {target_direction}")
            
            print(f"🎯 Centering target until GPT confirms it's centered...")
            center_result = self.center_target_until_centered(target_object)
            print(f"🎯 Centering result: {center_result}")
            
            print(f"🚀 Starting approach phase...")
            approach_result = self.approach_object_slowly(target_object, target_distance)
            print(f"🚀 Approach result: {approach_result}")
            
            stop_result = stop()
            self.navigation_active = False
            
            final_target = self._extract_target_from_vision(vision_analysis)
            return f"✅ Navigation complete! {approach_result}. Target identified as: {final_target}. {stop_result}"
            
        except Exception as e:
            self.navigation_active = False
            return f"Navigation error: {str(e)}"

    def get_status(self) -> str:
        status = {
            "navigation_active": self.navigation_active,
            "target_distance": self.target_distance,
            "target_object": self.target_object,
            "attempt_count": self.attempt_count,
            "locked_target": self.locked_target,
            "lock_confidence": self.lock_confidence
        }
        return f"Navigation Agent Status: {status}"

def main():
    agent = NavigationAgent()
    
    print("🤖 Navigation Agent Ready!")
    print("Commands:")
    print("  navigate [distance] [object] - Navigate to object at distance")
    print("  status - Get agent status")
    print("  quit - Exit")
    
    while True:
        try:
            command = input("\nNav> ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif command.lower() == 'status':
                print(agent.get_status())
            elif command.startswith('navigate'):
                parts = command.split()
                if len(parts) >= 3:
                    try:
                        distance = float(parts[1])
                        target = ' '.join(parts[2:])
                        print(f"🎯 Navigating to {target} at {distance}cm...")
                        result = agent.navigate_to_distance(distance, target)
                        print(f"Result: {result}")
                    except ValueError:
                        print("❌ Invalid distance. Use: navigate [distance] [object]")
                else:
                    print("❌ Usage: navigate [distance] [object]")
            else:
                print("❌ Unknown command. Use 'navigate', 'status', or 'quit'")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()