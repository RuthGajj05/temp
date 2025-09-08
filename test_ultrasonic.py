#!/usr/bin/env python3
"""
Test ultrasonic sensor readings
"""

import sys
import os
import time

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from primitives import get_ultrasonic_distance, get_picarx

def test_ultrasonic_sensor():
    """Test the ultrasonic sensor and diagnose issues."""
    print("🔧 Ultrasonic Sensor Test")
    print("=" * 30)
    
    print("Testing ultrasonic sensor readings...")
    print("Press Ctrl+C to stop")
    print()
    
    valid_readings = 0
    invalid_readings = 0
    
    try:
        while True:
            # Get distance reading
            distance_result = get_ultrasonic_distance()
            print(f"Raw result: {distance_result}")
            
            # Try to parse the distance
            try:
                # Try different parsing methods
                parts = distance_result.split()
                print(f"Split parts: {parts}")
                
                # Method 1: Look for number in the string
                distance = None
                for part in parts:
                    try:
                        val = float(part)
                        if 0 <= val <= 500:  # Reasonable range
                            distance = val
                            break
                    except ValueError:
                        continue
                
                if distance is not None:
                    print(f"✅ Valid reading: {distance}cm")
                    valid_readings += 1
                else:
                    print(f"❌ No valid distance found in: {distance_result}")
                    invalid_readings += 1
                    
            except Exception as e:
                print(f"❌ Error parsing: {e}")
                invalid_readings += 1
            
            print(f"Valid: {valid_readings}, Invalid: {invalid_readings}")
            print("-" * 30)
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n📊 Test Summary:")
        print(f"Valid readings: {valid_readings}")
        print(f"Invalid readings: {invalid_readings}")
        
        if invalid_readings > valid_readings:
            print("\n⚠️ Sensor appears to be malfunctioning!")
            print("Possible causes:")
            print("1. Ultrasonic sensor not connected properly")
            print("2. Sensor damaged")
            print("3. Power supply issues")
            print("4. Software compatibility issues")
        else:
            print("\n✅ Sensor appears to be working correctly!")

def test_ultrasonic_direct():
    """Test ultrasonic sensor directly using PicarX."""
    print("🔧 Direct Ultrasonic Sensor Test")
    print("=" * 35)
    
    try:
        px = get_picarx()
        print("PicarX instance created successfully")
        
        print("Testing ultrasonic.read() directly...")
        for i in range(10):
            try:
                distance = px.ultrasonic.read()
                print(f"Reading {i+1}: {distance}cm")
                
                if distance < 0:
                    print("  ⚠️ Negative reading detected!")
                elif distance > 500:
                    print("  ⚠️ Very large reading detected!")
                else:
                    print("  ✅ Reading looks valid")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
            
            time.sleep(0.5)
            
    except Exception as e:
        print(f"❌ Failed to create PicarX instance: {e}")

if __name__ == "__main__":
    print("Choose test method:")
    print("1. Test using primitives (get_ultrasonic_distance)")
    print("2. Test direct PicarX ultrasonic.read()")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        test_ultrasonic_sensor()
    elif choice == "2":
        test_ultrasonic_direct()
    else:
        print("Invalid choice. Running both tests...")
        test_ultrasonic_direct()
        print("\n" + "="*50 + "\n")
        test_ultrasonic_sensor()
