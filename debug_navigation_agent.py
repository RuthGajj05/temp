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
        
    def take_5_pictures(self, target_object: str = "nearest object") -> list:
        """Take 5 pictures from different camera angles."""
        try:
            # Initialize camera
            camera_result = init_camera()
            print(f"📷 Camera: {camera_result}")
            
            # Define camera positions for 5-frame panorama
            camera_positions = [
                {"pan": -20, "tilt": 0, "name": "left"},
                {"pan": 20, "tilt": 0, "name": "right"}, 
                {"pan": 0, "tilt": -10, "name": "down"},
                {"pan": 0, "tilt": 0, "name": "center"},
                {"pan": 0, "tilt": 10, "name": "up"}
            ]
            
            image_paths = []
            print(f"📸 Taking 5-frame panorama for '{target_object}'...")
            
            for i, pos in enumerate(camera_positions):
                # Move camera to position
                pan_result = set_cam_pan_angle(pos["pan"])
                tilt_result = set_cam_tilt_angle(pos["tilt"])
                print(f"📷 Position {i+1}/5: {pos['name']} (pan={pos['pan']}°, tilt={pos['tilt']}°)")
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
    
    def send_to_assistant(self, image_paths: list, target_object: str) -> str:
        """Send all 5 images to GPT-4 Vision assistant."""
        try:
            if len(image_paths) != 5:
                return f"❌ Expected 5 images, got {len(image_paths)}"
            
            # Prepare all images for the vision call
            view_names = ["left", "right", "down", "center", "up"]
            contents = []
            
            # Add instruction text
            contents.append({
                "type": "text",
                "text": f"You will receive 5 labeled views. Use them to pick the NEAREST {target_object} and tell me the best view."
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
            return analysis
            
        except Exception as e:
            return f"❌ Error sending to assistant: {str(e)}"
    
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
            for view in ["LEFT", "RIGHT", "DOWN", "CENTER", "UP"]:
                view_match = re.search(f'{view}:\s*([^\n]+)', response, re.IGNORECASE)
                if view_match:
                    parsed["view_analyses"][view.lower()] = view_match.group(1).strip()
                    print(f"🔍 {view}: {parsed['view_analyses'][view.lower()]}")
            
            return parsed
            
        except Exception as e:
            print(f"❌ Error parsing response: {str(e)}")
            return {"error": str(e)}
    
    def debug_navigation(self, target_object: str = "nearest object") -> dict:
        """Complete debug workflow: take pictures, send to assistant, parse response."""
        print("🔍 DEBUG NAVIGATION WORKFLOW")
        print("=" * 50)
        
        # Step 1: Take 5 pictures
        print("\n📸 STEP 1: Taking 5 pictures...")
        image_paths = self.take_5_pictures(target_object)
        
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
        
        print("\n📊 FINAL RESULTS:")
        print("=" * 30)
        for key, value in parsed.items():
            if key != "view_analyses":
                print(f"{key.upper()}: {value}")
        
        print("\nVIEW ANALYSES:")
        for view, analysis in parsed.get("view_analyses", {}).items():
            print(f"  {view.upper()}: {analysis}")
        
        return {
            "image_paths": image_paths,
            "raw_response": response,
            "parsed_data": parsed
        }

def main():
    """Interactive debug navigation agent."""
    print("🔍 PicarX Debug Navigation Agent")
    print("=" * 40)
    print("This agent will:")
    print("1. Take 5 pictures from different angles")
    print("2. Send them to GPT-4 Vision")
    print("3. Parse the response")
    print("4. Show you the results")
    print("NO MOVEMENT - DEBUGGING ONLY")
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
                print("- 'debug' - Run full debug workflow")
                print("- 'debug nearest object' - Debug with specific target")
                print("- 'quit' - Exit")
            elif command.lower() == "debug":
                result = agent.debug_navigation()
                if "error" in result:
                    print(f"❌ Error: {result['error']}")
                else:
                    print("✅ Debug workflow completed successfully!")
            elif command.lower().startswith("debug "):
                target = command[6:].strip()
                result = agent.debug_navigation(target)
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
