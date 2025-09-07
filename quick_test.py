#!/usr/bin/env python3
"""
Quick Interactive Test for PicarX Robot Primitives
Simple script for testing individual functions
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from primitives import *

def interactive_menu():
    """Interactive menu for testing primitives."""
    print("🤖 PicarX Robot Primitives - Quick Test")
    print("=" * 40)
    
    while True:
        print("\nSelect a test:")
        print("1.  Reset robot")
        print("2.  Move forward (2s)")
        print("3.  Move backward (1s)")
        print("4.  Turn left (20°, 2s)")
        print("5.  Turn right (20°, 2s)")
        print("6.  Set direction servo (15°)")
        print("7.  Set camera pan (10°)")
        print("8.  Set camera tilt (-10°)")
        print("9.  Read ultrasonic sensor")
        print("10. Read grayscale sensors")
        print("11. Initialize camera")
        print("12. Capture image")
        print("13. Get robot status")
        print("14. Test continuous movement")
        print("15. Stop robot")
        print("0.  Exit")
        
        try:
            choice = input("\nEnter choice (0-15): ").strip()
            
            if choice == "0":
                print("👋 Goodbye!")
                break
            elif choice == "1":
                result = reset()
                print(f"Result: {result}")
            elif choice == "2":
                result = move_forward(50, 2.0)
                print(f"Result: {result}")
            elif choice == "3":
                result = move_backward(40, 1.0)
                print(f"Result: {result}")
            elif choice == "4":
                result = turn_left(20, 50, 2.0)
                print(f"Result: {result}")
            elif choice == "5":
                result = turn_right(20, 50, 2.0)
                print(f"Result: {result}")
            elif choice == "6":
                result = set_dir_servo_angle(15)
                print(f"Result: {result}")
            elif choice == "7":
                result = set_cam_pan_angle(10)
                print(f"Result: {result}")
            elif choice == "8":
                result = set_cam_tilt_angle(-10)
                print(f"Result: {result}")
            elif choice == "9":
                result = get_ultrasonic_distance()
                print(f"Result: {result}")
            elif choice == "10":
                result = get_grayscale_data()
                print(f"Result: {result}")
            elif choice == "11":
                result = init_camera()
                print(f"Result: {result}")
            elif choice == "12":
                result = capture_image("quick_test.jpg")
                print(f"Result: {result}")
            elif choice == "13":
                result = get_robot_status()
                print(f"Result: {result}")
            elif choice == "14":
                print("Starting continuous movement (press Ctrl+C to stop)...")
                result = move_forward(30, None)
                print(f"Result: {result}")
                try:
                    input("Press Enter to stop...")
                except KeyboardInterrupt:
                    pass
                result = stop()
                print(f"Stop result: {result}")
            elif choice == "15":
                result = stop()
                print(f"Result: {result}")
            else:
                print("❌ Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            print("🛑 Stopping robot...")
            stop()
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Final cleanup
    print("\n🔄 Final robot reset...")
    reset()

if __name__ == "__main__":
    interactive_menu()
