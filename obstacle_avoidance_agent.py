#!/usr/bin/env python3

import sys
import os
import time
from typing import Optional, List, Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from primitives import *
from openai import OpenAI
from keys import OPENAI_API_KEY
import base64

class ObstacleAvoidanceAgent:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.navigation_active = False
        self.target_object = None
        self.max_attempts = 50
        self.attempt_count = 0
        self.target_location = None
        
    def analyze_surroundings(self, target_object: str, estimate_distance: bool = False, panoramic: bool = False) -> str:
        try:
            camera_result = init_camera()
            print(f"Camera: {camera_result}")
            
            print("📸 Taking 3-frame panorama (left, center, right) for obstacle analysis...")
            photo_results = []
            
            camera_positions = [
                {"pan": -20, "tilt": 0, "name": "left"},
                {"pan": 0, "tilt": 0, "name": "center"}, 
                {"pan": 20, "tilt": 0, "name": "right"}
            ]
            
            for pos in camera_positions:
                set_cam_pan_angle(pos["pan"])
                set_cam_tilt_angle(pos["tilt"])
                time.sleep(0.5)
                
                photo_result = capture_image(f"obstacle_{pos['name']}.jpg")
                photo_results.append(photo_result)
                print(f"📸 {pos['name']}: {photo_result}")
            
            set_cam_pan_angle(0)
            set_cam_tilt_angle(0)
            
            return self._analyze_panoramic_obstacles(photo_results, target_object, estimate_distance)
                
        except Exception as e:
            return f"Obstacle analysis failed: {str(e)}"

    def _analyze_panoramic_obstacles(self, photo_results: list, target_object: str, estimate_distance: bool = False) -> str:
        try:
            image_paths = []
            for result in photo_results:
                import re
                path_match = re.search(r': (.+\.jpg)', result)
                if path_match:
                    image_paths.append(path_match.group(1))
            
            if not image_paths:
                return "❌ No valid images captured for obstacle analysis"
            
            print(f"🔍 Analyzing {len(image_paths)} panorama images for obstacles and '{target_object}'...")
            
            for attempt in range(3):
                try:
                    analysis = self._send_obstacle_analysis_to_assistant(image_paths, target_object, estimate_distance)
                    
                    if self._is_valid_obstacle_response(analysis):
                        print(f"✅ Valid obstacle analysis received on attempt {attempt + 1}")
                        return analysis
                    else:
                        print(f"⚠️ Invalid obstacle response on attempt {attempt + 1}, retrying...")
                        time.sleep(1)
                        
                except Exception as e:
                    print(f"⚠️ Obstacle analysis attempt {attempt + 1} failed: {str(e)}")
                    if attempt < 2:
                        time.sleep(1)
                    else:
                        return f"❌ Obstacle analysis failed after 3 attempts: {str(e)}"
            
            return "❌ Obstacle analysis failed after 3 attempts"
            
        except Exception as e:
            return f"Panoramic obstacle analysis error: {str(e)}"

    def _is_valid_obstacle_response(self, response: str) -> bool:
        required_fields = ["TARGET:", "OBSTACLES:", "CLEAR_PATH:", "RECOMMENDATION:"]
        
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

    def _send_obstacle_analysis_to_assistant(self, image_paths: list, target_object: str, estimate_distance: bool = False) -> str:
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
                distance_instruction = " Also estimate distances to obstacles and target."
            
            prompt = f"""TASK: Navigate to the target '{target_object}' while avoiding obstacles in the path.

Analyze these 3 images (left, center, right) to find the target and identify obstacles blocking the path.

For each image, provide:
- TARGET: Description of target object if visible
- OBSTACLES: List of obstacles and their positions
- CLEAR_PATH: Direction with clearest path (left/center/right)
- RECOMMENDATION: Suggested action (move_forward/turn_left/turn_right/stop)

Format your response exactly as:
TARGET: [description or "not visible"]
OBSTACLES: [list of obstacles with positions]
CLEAR_PATH: [left/center/right]
RECOMMENDATION: [action]{distance_instruction}

Remember: The goal is to reach '{target_object}' while safely navigating around any obstacles."""

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
            return f"Obstacle analysis error: {str(e)}"


    def _parse_obstacle_analysis(self, analysis: str) -> dict:
        try:
            lines = analysis.split('\n')
            result = {}
            
            for line in lines:
                line = line.strip()
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().upper()
                    value = value.strip()
                    
                    if key == "TARGET":
                        result["target"] = value
                    elif key == "OBSTACLES":
                        result["obstacles"] = value
                    elif key == "CLEAR_PATH":
                        result["clear_path"] = value.lower()
                    elif key == "RECOMMENDATION":
                        result["recommendation"] = value.lower()
            
            return result
            
        except Exception as e:
            print(f"Error parsing obstacle analysis: {str(e)}")
            return {}

    def _get_stable_distance(self, samples: int = 5) -> Optional[float]:
        try:
            import re
            vals = []
            for _ in range(samples):
                s = get_ultrasonic_distance()
                m = re.search(r'([-+]?\d+(?:\.\d+)?)\s*cm', s)
                if m:
                    d = float(m.group(1))
                    if 2.0 <= d <= 400.0:
                        vals.append(d)
                time.sleep(0.06)
            
            if not vals:
                return None
            
            vals.sort()
            return vals[len(vals)//2]
            
        except Exception as e:
            print(f"Error getting stable distance: {str(e)}")
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

    def _determine_movement_strategy(self, analysis: str) -> str:
        try:
            parsed = self._parse_obstacle_analysis(analysis)
            
            if not parsed:
                return "stop"
            
            recommendation = parsed.get("recommendation", "stop")
            clear_path = parsed.get("clear_path", "center")
            
            print(f"📊 Obstacle Analysis: {parsed}")
            print(f"📊 Clear path: {clear_path}")
            print(f"📊 Recommendation: {recommendation}")
            
            if recommendation == "move_forward":
                return "forward"
            elif recommendation == "turn_left":
                return "turn_left"
            elif recommendation == "turn_right":
                return "turn_right"
            else:
                return "stop"
                
        except Exception as e:
            print(f"Error determining movement strategy: {str(e)}")
            return "stop"

    def navigate_around_obstacles(self, target_object: str) -> str:
        try:
            self.target_object = target_object
            self.navigation_active = True
            self.attempt_count = 0
            
            reset_result = reset()
            print(f"Reset: {reset_result}")
            
            print(f"🎯 OBSTACLE AVOIDANCE: Navigating to '{target_object}' while avoiding obstacles (using 3-frame panorama for each step)")
            
            while self.attempt_count < self.max_attempts:
                self.attempt_count += 1
                print(f"\n🔄 Attempt {self.attempt_count}/{self.max_attempts}")
                
                print(f"🔍 Analyzing surroundings for obstacles while navigating to '{target_object}'...")
                analysis = self.analyze_surroundings(target_object, estimate_distance=True, panoramic=True)
                print(f"Analysis: {analysis}")
                
                if "error" in analysis.lower() or "failed" in analysis.lower():
                    print(f"⚠️ Analysis failed on attempt {self.attempt_count}")
                    if self.attempt_count < self.max_attempts:
                        time.sleep(1)
                        continue
                    else:
                        self.navigation_active = False
                        return f"❌ Obstacle avoidance failed: Could not analyze surroundings after {self.max_attempts} attempts"
                
                strategy = self._determine_movement_strategy(analysis)
                print(f"📊 Movement strategy: {strategy}")
                
                if strategy == "forward":
                    print("🚀 Moving forward...")
                    move_result = move_forward(40, 1.0, check_obstacles=True)
                    print(f"Move result: {move_result}")
                    
                    if "obstacle detected" in move_result.lower():
                        print("⚠️ Obstacle detected during forward movement")
                        print("🔄 Taking evasive action...")
                        turn_result = turn_in_place_right(45, 50, 1.0)
                        print(f"Turn result: {turn_result}")
                        time.sleep(0.5)
                    
                elif strategy == "turn_left":
                    print("🔄 Turning left...")
                    turn_result = turn_in_place_left(30, 50, 1.0)
                    print(f"Turn result: {turn_result}")
                    time.sleep(0.5)
                    
                elif strategy == "turn_right":
                    print("🔄 Turning right...")
                    turn_result = turn_in_place_right(30, 50, 1.0)
                    print(f"Turn result: {turn_result}")
                    time.sleep(0.5)
                    
                else:
                    print("⏹️ Stopping navigation")
                    self.navigation_active = False
                    stop_result = stop()
                    return f"✅ Navigation stopped. {stop_result}"
                
                time.sleep(0.5)
            
            self.navigation_active = False
            stop_result = stop()
            return f"⚠️ Navigation incomplete after {self.max_attempts} attempts. {stop_result}"
            
        except Exception as e:
            self.navigation_active = False
            return f"Obstacle avoidance error: {str(e)}"

    def get_status(self) -> str:
        status = {
            "navigation_active": self.navigation_active,
            "target_object": self.target_object,
            "attempt_count": self.attempt_count,
            "max_attempts": self.max_attempts
        }
        return f"Obstacle Avoidance Agent Status: {status}"

def main():
    agent = ObstacleAvoidanceAgent()
    
    print("🤖 Obstacle Avoidance Agent Ready!")
    print("Commands:")
    print("  navigate [object] - Navigate to object while avoiding obstacles")
    print("  status - Get agent status")
    print("  quit - Exit")
    
    while True:
        try:
            command = input("\nObstacle> ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif command.lower() == 'status':
                print(agent.get_status())
            elif command.startswith('navigate'):
                parts = command.split()
                if len(parts) >= 2:
                    target = ' '.join(parts[1:])
                    print(f"🎯 Navigating to {target} while avoiding obstacles...")
                    result = agent.navigate_around_obstacles(target)
                    print(f"Result: {result}")
                else:
                    print("❌ Usage: navigate [object]")
            else:
                print("❌ Unknown command. Use 'navigate', 'status', or 'quit'")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()