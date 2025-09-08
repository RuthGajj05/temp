#!/usr/bin/env python3
"""
Test the navigation fix
"""

import sys
import os
import time

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from navigation_agent import NavigationAgent

def test_stable_distance():
    """Test the stable distance reading method."""
    print("🧪 Testing Stable Distance Reading")
    print("=" * 40)
    
    agent = NavigationAgent()
    
    print("Testing _get_stable_distance method...")
    for i in range(10):
        distance = agent._get_stable_distance()
        if distance is not None:
            print(f"✅ Reading {i+1}: {distance}cm")
        else:
            print(f"❌ Reading {i+1}: Failed to get valid distance")
        time.sleep(0.5)

def test_navigation_with_fix():
    """Test navigation with the fix applied."""
    print("🧪 Testing Navigation with Fix")
    print("=" * 35)
    
    agent = NavigationAgent()
    
    print("Testing navigation to 100cm from object...")
    print("This should now handle invalid readings gracefully.")
    print()
    
    result = agent.navigate_to_distance(100.0, "test object")
    print(f"\nResult: {result}")

if __name__ == "__main__":
    print("Choose test:")
    print("1. Test stable distance reading")
    print("2. Test full navigation")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        test_stable_distance()
    elif choice == "2":
        test_navigation_with_fix()
    else:
        print("Running both tests...")
        test_stable_distance()
        print("\n" + "="*50 + "\n")
        test_navigation_with_fix()
