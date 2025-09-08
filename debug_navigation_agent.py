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
                return f"✅ Turned left: {turn_result}"
                
            elif best_view == "right":
                print("🔄 Turning right towards object...")
                turn_result = turn_in_place_right(30, 20, 1.0)  # 30 degrees, 20% speed, 1 second
                return f"✅ Turned right: {turn_result}"
                
            elif best_view == "center":
                print("✅ Object already centered, no turning needed")
                return "✅ Object already centered"
                
            else:
                return f"❌ Unknown best view: {best_view}"
                
        except Exception as e:
            return f"❌ Error turning towards object: {str(e)}"
    
    def debug_navigation(self, target_object: str = "nearest object", turn_towards: bool = True) -> dict:
        """Complete debug workflow: take pictures, send to assistant, parse response, optionally turn."""
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
        
        return {
            "image_paths": image_paths,
            "raw_response": response,
            "parsed_data": parsed,
            "turn_result": turn_result
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
    print("5. Show you the results")
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
                
                if remaining == "no-turn":
                    # Just "debug no-turn" - use default target
                    result = agent.debug_navigation(turn_towards=False)
                elif remaining.endswith(" no-turn"):
                    # Target + no-turn
                    target = remaining[:-8].strip()  # Remove " no-turn"
                    result = agent.debug_navigation(target, turn_towards=False)
                else:
                    # Just target (could be multiple words like "nearest object")
                    target = remaining
                    result = agent.debug_navigation(target, turn_towards=True)
                
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
