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
        
    def analyze_surroundings(self, target_object: str) -> str:
        """Analyze the surroundings to identify the target object and obstacles."""
        try:
            # Initialize camera
            camera_result = init_camera()
            print(f"Camera: {camera_result}")
            
            # Take a picture of the surroundings
            photo_result = capture_image("obstacle_analysis.jpg")
            print(f"Photo: {photo_result}")
            
            # Extract file path from result
            if "successfully" in photo_result.lower():
                # Find the file path in the result
                import re
                path_match = re.search(r': (.+\.jpg)', photo_result)
                if path_match:
                    image_path = path_match.group(1)
                    
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
                                        "text": f"Look at this image and identify: 1) Can you see a {target_object}? If yes, describe its location (left, right, center, far, close). 2) What obstacles are visible that might block navigation? 3) What is the best path to reach the {target_object}? Describe the navigation strategy."
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
            vision_analysis = self.analyze_surroundings(target_object)
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
                    current_distance = float(distance_result.split()[4])
                except (IndexError, ValueError):
                    current_distance = 999.0  # Assume far if can't read
                
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
                
                # Take a photo for visual analysis
                photo_result = capture_image(f"navigation_attempt_{self.attempt_count}.jpg")
                print(f"Photo: {photo_result}")
                
                # Re-analyze surroundings periodically for better navigation
                if self.attempt_count % 3 == 0:  # Every 3 attempts
                    print("🔄 Re-analyzing surroundings for better navigation...")
                    vision_analysis = self.analyze_surroundings(target_object)
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
