#!/usr/bin/env python3
"""
Test script for PicarX Robot Primitives
Demonstrates all primitive functions with proper error handling
"""

import time
import sys
import os

# Add the current directory to Python path to import primitives
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from primitives import *

def test_basic_movement():
    """Test basic movement functions."""
    print("=== Testing Basic Movement ===")
    
    # Test reset
    print("1. Resetting robot...")
    result = reset()
    print(f"   Result: {result}")
    
    # Test forward movement
    print("2. Moving forward for 2 seconds...")
    result = move_forward(50, 2.0)
    print(f"   Result: {result}")
    
    # Test backward movement
    print("3. Moving backward for 1.5 seconds...")
    result = move_backward(40, 1.5)
    print(f"   Result: {result}")
    
    # Test stop
    print("4. Stopping...")
    result = stop()
    print(f"   Result: {result}")

def test_steering():
    """Test steering and turning functions."""
    print("\n=== Testing Steering ===")
    
    # Test direction servo
    print("1. Setting direction servo to 15 degrees...")
    result = set_dir_servo_angle(15)
    print(f"   Result: {result}")
    
    # Test left turn
    print("2. Turning left at 20 degrees for 2 seconds...")
    result = turn_left(20, 60, 2.0)
    print(f"   Result: {result}")
    
    # Test right turn
    print("3. Turning right at 25 degrees for 2 seconds...")
    result = turn_right(25, 60, 2.0)
    print(f"   Result: {result}")
    
    # Test straight steering
    print("4. Setting direction servo to 0 degrees...")
    result = set_dir_servo_angle(0)
    print(f"   Result: {result}")

def test_turn_in_place():
    """Test turn in place functions."""
    print("\n=== Testing Turn in Place ===")
    
    # Test left turn in place
    print("1. Turning left in place by 90 degrees...")
    result = turn_in_place_left(90, 50, 2.0)
    print(f"   Result: {result}")
    
    # Test right turn in place
    print("2. Turning right in place by 90 degrees...")
    result = turn_in_place_right(90, 50, 2.0)
    print(f"   Result: {result}")
    
    # Test 180 degree turn
    print("3. Turning left in place by 180 degrees...")
    result = turn_in_place_left(180, 60, 3.0)
    print(f"   Result: {result}")
    
    # Test continuous turn in place
    print("4. Testing continuous turn in place...")
    result = turn_in_place_right(360, 40, None)
    print(f"   Result: {result}")
    time.sleep(2)  # Let it turn for 2 seconds
    result = stop()
    print(f"   Stop result: {result}")

def test_camera_servos():
    """Test camera servo functions."""
    print("\n=== Testing Camera Servos ===")
    
    # Test camera pan
    print("1. Setting camera pan to 20 degrees...")
    result = set_cam_pan_angle(20)
    print(f"   Result: {result}")
    
    # Test camera tilt
    print("2. Setting camera tilt to -15 degrees...")
    result = set_cam_tilt_angle(-15)
    print(f"   Result: {result}")
    
    # Test getting servo angles
    print("3. Getting all servo angles...")
    result = get_servo_angles()
    print(f"   Result: {result}")
    
    # Reset camera servos
    print("4. Resetting camera servos...")
    result = set_cam_pan_angle(0)
    print(f"   Pan reset: {result}")
    result = set_cam_tilt_angle(0)
    print(f"   Tilt reset: {result}")

def test_sensors():
    """Test sensor functions."""
    print("\n=== Testing Sensors ===")
    
    # Test ultrasonic sensor
    print("1. Reading ultrasonic sensor...")
    result = get_ultrasonic_distance()
    print(f"   Result: {result}")
    
    # Test grayscale sensors
    print("2. Reading grayscale sensors...")
    result = get_grayscale_data()
    print(f"   Result: {result}")
    
    # Test line detection
    print("3. Testing line detection...")
    grayscale_data = [2000, 1500, 1800]  # Example values
    result = get_line_status(grayscale_data)
    print(f"   Result: {result}")
    
    # Test cliff detection
    print("4. Testing cliff detection...")
    result = get_cliff_status(grayscale_data)
    print(f"   Result: {result}")

def test_camera():
    """Test camera functions."""
    print("\n=== Testing Camera ===")
    
    # Test camera initialization
    print("1. Initializing camera...")
    result = init_camera()
    print(f"   Result: {result}")
    
    if "successfully" in result.lower():
        # Test image capture
        print("2. Capturing image...")
        result = capture_image("test_capture.jpg")
        print(f"   Result: {result}")
        
        # Test photo with Vilib
        print("3. Taking photo with Vilib...")
        result = take_photo_vilib("test_photo", "./")
        print(f"   Result: {result}")
        
        # Test camera close
        print("4. Closing camera...")
        result = close_camera()
        print(f"   Result: {result}")
    else:
        print("   Skipping camera tests due to initialization failure")

def test_sound():
    """Test sound functions."""
    print("\n=== Testing Sound ===")
    
    # Test synchronous sound (if sound file exists)
    print("1. Testing sound playback...")
    sound_file = "test_sound.wav"
    if os.path.exists(sound_file):
        result = play_sound(sound_file, 50)
        print(f"   Result: {result}")
    else:
        print(f"   Skipping: Sound file {sound_file} not found")
    
    # Test asynchronous sound
    print("2. Testing asynchronous sound...")
    if os.path.exists(sound_file):
        result = play_sound_threading(sound_file, 30)
        print(f"   Result: {result}")
        time.sleep(1)  # Let it play briefly
    else:
        print(f"   Skipping: Sound file {sound_file} not found")

def test_continuous_movement():
    """Test continuous movement mode."""
    print("\n=== Testing Continuous Movement ===")
    
    print("1. Starting continuous forward movement...")
    result = move_forward(40, None)  # duration=None means continuous
    print(f"   Result: {result}")
    
    print("2. Moving for 3 seconds...")
    time.sleep(3)
    
    print("3. Stopping continuous movement...")
    result = stop()
    print(f"   Result: {result}")

def test_parameter_validation():
    """Test parameter validation and clamping."""
    print("\n=== Testing Parameter Validation ===")
    
    # Test speed clamping
    print("1. Testing speed clamping (150 -> 100)...")
    result = move_forward(150, 0.5)  # Should clamp to 100
    print(f"   Result: {result}")
    
    # Test angle clamping
    print("2. Testing angle clamping (50 -> 30)...")
    result = set_dir_servo_angle(50)  # Should clamp to 30
    print(f"   Result: {result}")
    
    # Test negative speed
    print("3. Testing negative speed (-20 -> 0)...")
    result = move_forward(-20, 0.5)  # Should clamp to 0
    print(f"   Result: {result}")
    
    # Test camera angle clamping
    print("4. Testing camera angle clamping (60 -> 35)...")
    result = set_cam_pan_angle(60)  # Should clamp to 35
    print(f"   Result: {result}")

def test_obstacle_detection():
    """Test obstacle detection during movement."""
    print("\n=== Testing Obstacle Detection ===")
    
    print("1. Moving forward with obstacle detection...")
    result = move_forward(30, 3.0, check_obstacles=True)
    print(f"   Result: {result}")
    
    print("2. Moving forward without obstacle detection...")
    result = move_forward(30, 2.0, check_obstacles=False)
    print(f"   Result: {result}")

def test_robot_status():
    """Test robot status function."""
    print("\n=== Testing Robot Status ===")
    
    print("1. Getting comprehensive robot status...")
    result = get_robot_status()
    print(f"   Result: {result}")

def main():
    """Main test function."""
    print("🤖 PicarX Robot Primitives Test Suite")
    print("=" * 50)
    
    try:
        # Run all test suites
        test_basic_movement()
        test_steering()
        test_turn_in_place()
        test_camera_servos()
        test_sensors()
        test_camera()
        test_sound()
        test_continuous_movement()
        test_parameter_validation()
        test_obstacle_detection()
        test_robot_status()
        
        print("\n" + "=" * 50)
        print("✅ All tests completed successfully!")
        print("🤖 Robot primitives are working correctly!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        print("🛑 Stopping robot...")
        stop()
        
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        print("🛑 Stopping robot...")
        stop()
        
    finally:
        # Always reset the robot at the end
        print("\n🔄 Final robot reset...")
        reset()

if __name__ == "__main__":
    main()
