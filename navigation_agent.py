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
        
    def analyze_surroundings(self, target_object: str) -> str:
        """Analyze the surroundings to identify the target object."""
        try:
            # Initialize camera
            camera_result = init_camera()
            print(f"Camera: {camera_result}")
            
            # Take a picture of the surroundings
            photo_result = capture_image("surroundings_analysis.jpg")
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
                                        "text": f"Look at this image and identify if you can see a {target_object}. If you can see it, describe its location relative to the camera (left, right, center, far, close, etc.). If you cannot see it, say 'not visible' and describe what objects you can see instead."
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
            print(f"🔍 Analyzing surroundings to find {target_object}...")
            vision_analysis = self.analyze_surroundings(target_object)
            print(f"Vision Analysis: {vision_analysis}")
            
            # Start navigation loop
            while self.navigation_active:
                # Get current distance
                distance_result = get_ultrasonic_distance()
                print(f"Distance check: {distance_result}")
                
                # Extract distance from the result string
                try:
                    current_distance = float(distance_result.split()[5])  # "Ultrasound detected an obstacle at X.X cm"
                except (IndexError, ValueError):
                    return f"Error reading distance: {distance_result}"
                
                # Check if we've reached the target distance
                if current_distance <= target_distance:
                    stop_result = stop()
                    self.navigation_active = False
                    return f"✅ Navigation complete! Reached {target_distance}cm from {target_object}. Final distance: {current_distance:.1f}cm. {stop_result}"
                
                # Check if we're too close (safety check)
                if current_distance < 5.0:
                    stop_result = stop()
                    self.navigation_active = False
                    return f"⚠️ Safety stop! Too close to {target_object} at {current_distance:.1f}cm. {stop_result}"
                
                # Move forward with obstacle checking
                move_result = move_forward(30, 0.5, check_obstacles=True)
                print(f"Movement: {move_result}")
                
                # Check if movement was stopped due to obstacles
                if "stopped early" in move_result.lower():
                    self.navigation_active = False
                    return f"❌ Navigation stopped due to obstacle detection: {move_result}"
                
                # Periodically re-analyze surroundings to ensure we're heading toward the right object
                if self.attempt_count % 5 == 0:  # Every 5 attempts
                    print("🔄 Re-analyzing surroundings...")
                    vision_analysis = self.analyze_surroundings(target_object)
                    print(f"Updated Vision Analysis: {vision_analysis}")
                    
                    # If target is not visible, try to turn to find it
                    if "not visible" in vision_analysis.lower():
                        print("Target not visible, turning to search...")
                        turn_result = turn_in_place_left(45, 25, 1.0)
                        print(f"Search turn: {turn_result}")
                
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
