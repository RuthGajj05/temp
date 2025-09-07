#!/usr/bin/env python3
"""
PicarX Robot Primitives Demo
Shows key features and capabilities
"""

import sys
import os
import time

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from primitives import *

def demo_movement():
    """Demo basic movement capabilities."""
    print("🚗 Movement Demo")
    print("-" * 20)
    
    # Reset robot
    print("Resetting robot...")
    print(f"  {reset()}")
    
    # Forward movement
    print("\nMoving forward...")
    print(f"  {move_forward(60, 2.0)}")
    
    # Turn left
    print("\nTurning left...")
    print(f"  {turn_left(25, 50, 1.5)}")
    
    # Turn right
    print("\nTurning right...")
    print(f"  {turn_right(25, 50, 1.5)}")
    
    # Turn in place left
    print("\nTurning in place left (90°)...")
    print(f"  {turn_in_place_left(90, 50, 2.0)}")
    
    # Turn in place right
    print("\nTurning in place right (90°)...")
    print(f"  {turn_in_place_right(90, 50, 2.0)}")
    
    # Stop
    print("\nStopping...")
    print(f"  {stop()}")

def demo_sensors():
    """Demo sensor capabilities."""
    print("\n📡 Sensor Demo")
    print("-" * 20)
    
    # Ultrasonic sensor
    print("Reading ultrasonic sensor...")
    print(f"  {get_ultrasonic_distance()}")
    
    # Grayscale sensors
    print("\nReading grayscale sensors...")
    print(f"  {get_grayscale_data()}")
    
    # Servo status
    print("\nGetting servo angles...")
    print(f"  {get_servo_angles()}")

def demo_camera():
    """Demo camera capabilities."""
    print("\n📷 Camera Demo")
    print("-" * 20)
    
    # Initialize camera
    print("Initializing camera...")
    result = init_camera()
    print(f"  {result}")
    
    if "successfully" in result.lower():
        # Capture image
        print("\nCapturing image...")
        result = capture_image("demo_image.jpg")
        print(f"  {result}")
        
        # Close camera
        print("\nClosing camera...")
        print(f"  {close_camera()}")
    else:
        print("  Camera initialization failed - skipping capture")

def demo_continuous_movement():
    """Demo continuous movement."""
    print("\n🔄 Continuous Movement Demo")
    print("-" * 30)
    
    print("Starting continuous movement...")
    print(f"  {move_forward(40, None)}")
    
    print("Moving for 3 seconds...")
    time.sleep(3)
    
    print("Stopping...")
    print(f"  {stop()}")

def demo_parameter_validation():
    """Demo parameter validation."""
    print("\n✅ Parameter Validation Demo")
    print("-" * 35)
    
    # Test speed clamping
    print("Testing speed clamping (150 -> 100)...")
    print(f"  {move_forward(150, 0.5)}")
    
    # Test angle clamping
    print("\nTesting angle clamping (50 -> 30)...")
    print(f"  {set_dir_servo_angle(50)}")
    
    # Test negative values
    print("\nTesting negative speed (-30 -> 0)...")
    print(f"  {move_forward(-30, 0.5)}")

def main():
    """Main demo function."""
    print("🤖 PicarX Robot Primitives Demo")
    print("=" * 40)
    print("This demo shows the key capabilities of the robot primitives.")
    print("Make sure your PicarX robot is connected and ready!")
    print()
    
    try:
        # Run all demos
        demo_movement()
        demo_sensors()
        demo_camera()
        demo_continuous_movement()
        demo_parameter_validation()
        
        print("\n" + "=" * 40)
        print("✅ Demo completed successfully!")
        print("🎉 All primitives are working correctly!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        
    finally:
        # Always reset the robot
        print("\n🔄 Final robot reset...")
        reset()
        print("👋 Demo finished!")

if __name__ == "__main__":
    main()
