#!/usr/bin/env python3
"""
Obstacle Avoidance Agent for PicarX Robot
Handles complex navigation with obstacle detection and avoidance
"""

import sys
import os
import time
from typing import Optional, List, Tuple

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from primitives import *
from openai import OpenAI
from keys import OPENAI_API_KEY
import base64

class ObstacleAvoidanceAgent:
    def __init__(self):
        """Initialize the Obstacle Avoidance Agent."""
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.navigation_active = False
        self.target_object = None
        self.max_attempts = 50  # Maximum navigation attempts
        self.attempt_count = 0
        self.target_location = None  # Store vision analysis of target location
        
    def analyze_surroundings(self, target_object: str, estimate_distance: bool = False, panoramic: bool = False) -> str:
        """Analyze the surroundings to identify the target object and obstacles."""
        try:
            # Initialize camera
            camera_result = init_camera()
            print(f"Camera: {camera_result}")
            
            if panoramic:
                # Use camera pan/tilt for panoramic obstacle analysis
                print("📸 Taking panoramic photos for obstacle analysis...")
                photo_results = []
                analysis_results = []
                
                # Define camera positions for panoramic view
                camera_positions = [
                    {"pan": -25, "tilt": 0, "name": "left"},
                    {"pan": 0, "tilt": 0, "name": "center"}, 
                    {"pan": 25, "tilt": 0, "name": "right"},
                    {"pan": 0, "tilt": -15, "name": "down"},
                    {"pan": 0, "tilt": 15, "name": "up"}
                ]
                
                for i, pos in enumerate(camera_positions):
                    # Move camera to position
                    pan_result = set_cam_pan_angle(pos["pan"])
                    tilt_result = set_cam_tilt_angle(pos["tilt"])
                    print(f"📷 Camera position {pos['name']}: pan={pos['pan']}°, tilt={pos['tilt']}°")
                    time.sleep(0.3)  # Wait for camera to settle
                    
                    # Take photo
                    photo_result = capture_image(f"obstacle_panoramic_{pos['name']}_{i+1}.jpg")
                    photo_results.append(photo_result)
                    print(f"Photo {pos['name']}: {photo_result}")
                    
                    # Quick analysis of this specific view
                    if "successfully" in photo_result.lower():
                        quick_analysis = self._quick_analyze_obstacles(photo_result, target_object, pos["name"])
                        analysis_results.append(quick_analysis)
                        print(f"Quick analysis {pos['name']}: {quick_analysis}")
                
                # Reset camera to center
                set_cam_pan_angle(0)
                set_cam_tilt_angle(0)
                print("📷 Camera reset to center position")
                
                # Use the center photo for main analysis
                photo_result = photo_results[1]  # Center photo
                
                # Combine all quick analyses for better context
                combined_analysis = self._combine_obstacle_analysis(analysis_results, target_object)
                print(f"🔍 Combined panoramic obstacle analysis: {combined_analysis}")
                
            else:
                # Take a single picture of the surroundings
                photo_result = capture_image("obstacle_analysis.jpg")
                print(f"Photo: {photo_result}")
            
            # Extract file path from result
            if "successfully" in photo_result.lower():
                # Find the file path in the result
                import re
                path_match = re.search(r': (.+\.jpg)', photo_result)
                if path_match:
                    image_path = path_match.group(1)
                    
                    # Create prompt based on whether we need distance estimation
                    if estimate_distance:
                        prompt_text = f"Look at this image and identify: 1) Can you see a {target_object}? If yes, describe its location (left, right, center, far, close) and estimate the distance in centimeters. 2) What obstacles are visible that might block navigation? 3) What is the best path to reach the {target_object}? Describe the navigation strategy. Be specific about distance estimates (e.g., 'about 50cm away', 'very close at 10-15cm', 'far away at 100cm+')."
                    else:
                        prompt_text = f"Look at this image and identify: 1) Can you see a {target_object}? If yes, describe its location (left, right, center, far, close). 2) What obstacles are visible that might block navigation? 3) What is the best path to reach the {target_object}? Describe the navigation strategy."
                    
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
                        max_tokens=400
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
    
    def _extract_distance_from_vision(self, vision_analysis: str) -> Optional[float]:
        """Extract distance estimate from vision analysis text."""
        try:
            import re
            
            # Look for distance patterns in the vision analysis
            distance_patterns = [
                r'(\d+)\s*cm',  # "50cm", "30 cm"
                r'about\s+(\d+)\s*cm',  # "about 50cm"
                r'approximately\s+(\d+)\s*cm',  # "approximately 50cm"
                r'(\d+)\s*centimeters',  # "50 centimeters"
                r'(\d+)\s*-\s*(\d+)\s*cm',  # "10-15cm" (take average)
                r'(\d+)\s*to\s*(\d+)\s*cm',  # "10 to 15cm" (take average)
            ]
            
            for pattern in distance_patterns:
                matches = re.findall(pattern, vision_analysis.lower())
                if matches:
                    if len(matches[0]) == 2:  # Range like "10-15cm"
                        try:
                            min_dist = float(matches[0][0])
                            max_dist = float(matches[0][1])
                            return (min_dist + max_dist) / 2  # Return average
                        except ValueError:
                            continue
                    else:  # Single distance
                        try:
                            return float(matches[0])
                        except ValueError:
                            continue
            
            # Look for qualitative distance descriptions
            if any(word in vision_analysis.lower() for word in ['very close', 'extremely close', 'right next to']):
                return 10.0  # Very close
            elif any(word in vision_analysis.lower() for word in ['close', 'near', 'nearby']):
                return 25.0  # Close
            elif any(word in vision_analysis.lower() for word in ['medium distance', 'moderate distance']):
                return 50.0  # Medium distance
            elif any(word in vision_analysis.lower() for word in ['far', 'distant', 'far away']):
                return 100.0  # Far
            
            return None
            
        except Exception as e:
            print(f"Error extracting distance from vision: {e}")
            return None
    
    def _quick_analyze_obstacles(self, photo_result: str, target_object: str, view_name: str) -> str:
        """Quickly analyze a single photo for obstacles and target."""
        try:
            import re
            path_match = re.search(r': (.+\.jpg)', photo_result)
            if not path_match:
                return f"No photo available for {view_name} view"
            
            image_path = path_match.group(1)
            
            # Quick obstacle analysis prompt
            prompt_text = f"""Quick obstacle analysis of {view_name} view: 
            1. Can you see a {target_object}? 
            2. What obstacles are visible (walls, furniture, objects)? 
            3. Is the path clear in this direction?
            Respond with: TARGET: [visible/not visible] OBSTACLES: [what obstacles] PATH: [clear/blocked]"""
            
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
                max_tokens=150
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Quick obstacle analysis error for {view_name}: {str(e)}"
    
    def _combine_obstacle_analysis(self, analysis_results: list, target_object: str) -> str:
        """Combine multiple obstacle view analyses into a comprehensive understanding."""
        try:
            # Find which views detected the target and obstacles
            target_views = []
            obstacle_views = []
            clear_paths = []
            
            for i, analysis in enumerate(analysis_results):
                view_names = ["left", "center", "right", "down", "up"]
                view_name = view_names[i] if i < len(view_names) else f"view_{i+1}"
                
                if "TARGET: visible" in analysis or "visible" in analysis.lower():
                    target_views.append(view_name)
                
                if "OBSTACLES:" in analysis and "none" not in analysis.lower():
                    obstacle_views.append(view_name)
                
                if "PATH: clear" in analysis.lower():
                    clear_paths.append(view_name)
            
            # Create combined analysis
            if target_views:
                if len(target_views) == 1:
                    position = target_views[0]
                    confidence = 8
                elif "center" in target_views:
                    position = "center"
                    confidence = 9
                else:
                    position = target_views[0]
                    confidence = 7
                
                combined = f"""TARGET: visible
POSITION: {position} (detected in {', '.join(target_views)} views)
OBSTACLES: Found in {len(obstacle_views)} views: {', '.join(obstacle_views) if obstacle_views else 'none detected'}
CLEAR_PATHS: {', '.join(clear_paths) if clear_paths else 'none'}
CONFIDENCE: {confidence}
DETAILS: Target {target_object} found in {len(target_views)} views. Obstacles in {len(obstacle_views)} views. Clear paths: {', '.join(clear_paths)}"""
            else:
                combined = f"""TARGET: not visible
POSITION: unknown
OBSTACLES: Found in {len(obstacle_views)} views: {', '.join(obstacle_views) if obstacle_views else 'none detected'}
CLEAR_PATHS: {', '.join(clear_paths) if clear_paths else 'none'}
CONFIDENCE: 6
DETAILS: Target {target_object} not found. Obstacles in {len(obstacle_views)} views. Clear paths: {', '.join(clear_paths)}"""
            
            return combined
            
        except Exception as e:
            return f"Obstacle analysis combination error: {str(e)}"

    def navigate_around_obstacles(self, target_object: str = "destination") -> str:
        """
        Navigate to a target while avoiding obstacles.
        
        Args:
            target_object: Description of the target destination
        """
        try:
            self.target_object = target_object
            self.navigation_active = True
            self.attempt_count = 0
            
            # Reset robot to starting position
            reset_result = reset()
            print(f"Reset: {reset_result}")
            
            # Analyze surroundings to identify target and obstacles
            print(f"🔍 Analyzing surroundings to find {target_object} and identify obstacles...")
            vision_analysis = self.analyze_surroundings(target_object, estimate_distance=True, panoramic=True)
            print(f"Vision Analysis: {vision_analysis}")
            self.target_location = vision_analysis
            
            while self.navigation_active and self.attempt_count < self.max_attempts:
                self.attempt_count += 1
                print(f"\n--- Navigation Attempt {self.attempt_count} ---")
                
                # Get current sensor readings
                distance_result = get_ultrasonic_distance()
                grayscale_result = get_grayscale_data()
                print(f"Sensors - Distance: {distance_result}, Grayscale: {grayscale_result}")
                
                # Extract distance
                try:
                    current_distance = float(distance_result.split()[5])
                    
                    # Check if we got an invalid reading (negative or too large)
                    if current_distance < 0 or current_distance > 500:
                        print(f"⚠️ Invalid ultrasonic reading: {current_distance}cm, using camera backup...")
                        
                        # Use camera as backup for distance estimation
                        vision_analysis = self.analyze_surroundings(target_object, estimate_distance=True)
                        print(f"Vision Analysis: {vision_analysis}")
                        
                        # Extract distance from vision
                        vision_distance = self._extract_distance_from_vision(vision_analysis)
                        if vision_distance is not None:
                            current_distance = vision_distance
                            print(f"📷 Camera distance estimate: {current_distance}cm")
                        else:
                            print("⚠️ Could not extract distance from vision, using fallback...")
                            current_distance = 50.0  # Safe fallback
                        
                except (IndexError, ValueError) as e:
                    print(f"⚠️ Error parsing distance: {distance_result}")
                    print(f"Error details: {e}")
                    print("Using camera backup...")
                    
                    # Use camera as backup for distance estimation
                    vision_analysis = self.analyze_surroundings(target_object, estimate_distance=True)
                    print(f"Vision Analysis: {vision_analysis}")
                    
                    # Extract distance from vision
                    vision_distance = self._extract_distance_from_vision(vision_analysis)
                    if vision_distance is not None:
                        current_distance = vision_distance
                        print(f"📷 Camera distance estimate: {current_distance}cm")
                    else:
                        print("⚠️ Could not extract distance from vision, using fallback...")
                        current_distance = 50.0  # Safe fallback
                
                # Check if we've reached a reasonable distance (success)
                if current_distance < 20.0 and current_distance > 5.0:
                    stop_result = stop()
                    self.navigation_active = False
                    return f"✅ Successfully reached {target_object}! Final distance: {current_distance:.1f}cm. {stop_result}"
                
                # Safety check - too close
                if current_distance < 5.0:
                    stop_result = stop()
                    self.navigation_active = False
                    return f"⚠️ Safety stop! Too close at {current_distance:.1f}cm. {stop_result}"
                
                # Take a photo for visual analysis (every attempt)
                photo_result = capture_image(f"obstacle_nav_attempt_{self.attempt_count}.jpg")
                print(f"📸 Photo: {photo_result}")
                
                # Re-analyze surroundings more frequently for better navigation
                if self.attempt_count % 2 == 0:  # Every 2 attempts (more frequent)
                    print("🔄 Re-analyzing surroundings for better navigation...")
                    vision_analysis = self.analyze_surroundings(target_object, estimate_distance=True)
                    print(f"Updated Vision Analysis: {vision_analysis}")
                    self.target_location = vision_analysis
                
                # Decide on movement strategy based on sensors and vision
                if current_distance > 50.0:
                    # Far away - move forward
                    print("Strategy: Moving forward (far from target)")
                    move_result = move_forward(40, 1.0, check_obstacles=True)
                elif current_distance > 20.0:
                    # Medium distance - careful forward movement
                    print("Strategy: Careful forward movement (medium distance)")
                    move_result = move_forward(25, 0.5, check_obstacles=True)
                else:
                    # Close - fine positioning
                    print("Strategy: Fine positioning (close to target)")
                    move_result = move_forward(15, 0.3, check_obstacles=True)
                
                print(f"Movement: {move_result}")
                
                # Check if movement was blocked by obstacles
                if "stopped early" in move_result.lower():
                    print("Obstacle detected! Attempting avoidance...")
                    avoidance_result = self._avoid_obstacle()
                    print(f"Avoidance: {avoidance_result}")
                    
                    if "failed" in avoidance_result.lower():
                        self.navigation_active = False
                        return f"❌ Navigation failed: Unable to avoid obstacles. {avoidance_result}"
                
                # Small delay for sensor stability
                time.sleep(0.2)
            
            # If we exit the loop due to max attempts
            self.navigation_active = False
            return f"⚠️ Navigation timeout after {self.max_attempts} attempts. May need manual intervention."
            
        except Exception as e:
            self.navigation_active = False
            return f"Navigation error: {str(e)}"
    
    def _avoid_obstacle(self) -> str:
        """Attempt to avoid an obstacle using various strategies with vision guidance."""
        try:
            print("Attempting obstacle avoidance with vision guidance...")
            
            # Take a photo to analyze the obstacle
            photo_result = capture_image("obstacle_avoidance.jpg")
            print(f"Obstacle photo: {photo_result}")
            
            # Analyze the obstacle and find the best avoidance path
            if "successfully" in photo_result.lower():
                import re
                path_match = re.search(r': (.+\.jpg)', photo_result)
                if path_match:
                    image_path = path_match.group(1)
                    
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
                                        "text": f"I'm blocked by an obstacle. Look at this image and suggest the best direction to turn (left or right) to avoid the obstacle and continue toward the {self.target_object}. Consider the obstacle's position and the best path around it."
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
                        max_tokens=200
                    )
                    
                    avoidance_advice = response.choices[0].message.content
                    print(f"Vision Avoidance Advice: {avoidance_advice}")
                    
                    # Use vision advice to choose avoidance direction
                    if "left" in avoidance_advice.lower():
                        print("Strategy: Vision suggests turning left")
                        turn_result = turn_in_place_left(45, 30, 1.0)
                        print(f"Turn left: {turn_result}")
                    elif "right" in avoidance_advice.lower():
                        print("Strategy: Vision suggests turning right")
                        turn_result = turn_in_place_right(45, 30, 1.0)
                        print(f"Turn right: {turn_result}")
                    else:
                        # Default to left if unclear
                        print("Strategy: Default to turning left")
                        turn_result = turn_in_place_left(45, 30, 1.0)
                        print(f"Turn left: {turn_result}")
                else:
                    # Fallback to default strategy
                    print("Strategy: Fallback to default left turn")
                    turn_result = turn_in_place_left(45, 30, 1.0)
                    print(f"Turn left: {turn_result}")
            else:
                # Fallback to default strategy
                print("Strategy: Fallback to default left turn")
                turn_result = turn_in_place_left(45, 30, 1.0)
                print(f"Turn left: {turn_result}")
            
            # Check if path is clear after turning
            distance_result = get_ultrasonic_distance()
            try:
                distance = float(distance_result.split()[4])
                if distance > 20.0:
                    move_result = move_forward(25, 1.0, check_obstacles=True)
                    if "stopped early" not in move_result.lower():
                        return f"✅ Avoidance successful! Used vision guidance and moved forward. {move_result}"
            except (IndexError, ValueError):
                pass
            
            # If left turn didn't work, try right
            print("Strategy: Trying right turn")
            turn_result = turn_in_place_right(90, 30, 1.5)
            print(f"Turn right: {turn_result}")
            
            # Check if path is clear
            distance_result = get_ultrasonic_distance()
            try:
                distance = float(distance_result.split()[4])
                if distance > 20.0:
                    move_result = move_forward(25, 1.0, check_obstacles=True)
                    if "stopped early" not in move_result.lower():
                        return f"✅ Avoidance successful! Turned right and moved forward. {move_result}"
            except (IndexError, ValueError):
                pass
            
            # Strategy 3: Back up and try different angle
            print("Strategy 3: Back up and try different approach")
            backup_result = move_backward(20, 1.0)
            print(f"Backup: {backup_result}")
            
            turn_result = turn_in_place_left(60, 30, 1.0)
            print(f"Turn: {turn_result}")
            
            return "⚠️ Avoidance attempted but path may still be blocked"
            
        except Exception as e:
            return f"Avoidance error: {str(e)}"
    
    def stop_navigation(self) -> str:
        """Stop the current navigation task."""
        self.navigation_active = False
        stop_result = stop()
        return f"Obstacle avoidance navigation stopped. {stop_result}"
    
    def get_navigation_status(self) -> str:
        """Get current navigation status."""
        if not self.navigation_active:
            return "No active obstacle avoidance navigation task"
        
        try:
            distance_result = get_ultrasonic_distance()
            return f"Navigating around obstacles to {self.target_object} (attempt {self.attempt_count}/{self.max_attempts}). Current: {distance_result}"
        except Exception as e:
            return f"Status check error: {str(e)}"

def main():
    """Interactive obstacle avoidance agent."""
    print("🤖 PicarX Obstacle Avoidance Agent")
    print("=" * 45)
    print("Commands:")
    print("- 'navigate to red ball' - Navigate to target while avoiding obstacles")
    print("- 'navigate to couch' - Navigate to target while avoiding obstacles")
    print("- 'status' - Check current navigation status")
    print("- 'stop' - Stop current navigation")
    print("- 'quit' - Exit")
    print()
    
    agent = ObstacleAvoidanceAgent()
    
    while True:
        try:
            command = input("ObstacleAvoidance> ").strip().lower()
            
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
                # Extract target object
                target_object = command.replace("navigate to", "").strip()
                if not target_object:
                    target_object = "destination"
                
                print(f"🚗 Starting obstacle avoidance navigation to {target_object}...")
                result = agent.navigate_around_obstacles(target_object)
                print(f"Result: {result}")
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
