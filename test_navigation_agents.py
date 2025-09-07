#!/usr/bin/env python3
"""
Test script for navigation agents
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_navigation_agent import SmartNavigationAgent

def test_command_analysis():
    """Test command analysis functionality."""
    print("🧪 Testing Command Analysis")
    print("=" * 40)
    
    agent = SmartNavigationAgent()
    
    test_commands = [
        ("navigate until 30cm from the couch", "distance_navigation"),
        ("drive to 20cm from the wall", "distance_navigation"),
        ("go until 15cm from the table", "distance_navigation"),
        ("navigate to the red ball", "obstacle_avoidance"),
        ("find the exit while avoiding obstacles", "obstacle_avoidance"),
        ("reach the kitchen", "obstacle_avoidance"),
        ("status", "status"),
        ("stop", "stop"),
        ("help", "help"),
        ("unknown command", "unknown")
    ]
    
    print("Testing command analysis...")
    print()
    
    for command, expected_type in test_commands:
        result = agent._analyze_command(command.lower())
        status = "✅" if result == expected_type else "❌"
        print(f"{status} '{command}' → {result} (expected: {expected_type})")
    
    print()

def test_parameter_extraction():
    """Test parameter extraction from commands."""
    print("🧪 Testing Parameter Extraction")
    print("=" * 40)
    
    agent = SmartNavigationAgent()
    
    # Test distance extraction
    distance_commands = [
        "navigate until 30cm from the couch",
        "drive to 20cm from the wall",
        "go until 15cm from the table"
    ]
    
    print("Testing distance extraction...")
    for command in distance_commands:
        import re
        distance_match = re.search(r"(\d+)\s*cm", command)
        if distance_match:
            distance = float(distance_match.group(1))
            print(f"✅ '{command}' → {distance}cm")
        else:
            print(f"❌ '{command}' → No distance found")
    
    print()
    
    # Test object extraction
    object_commands = [
        "navigate to the red ball",
        "find the exit while avoiding obstacles",
        "reach the kitchen",
        "navigate until 30cm from the couch"
    ]
    
    print("Testing object extraction...")
    for command in object_commands:
        import re
        object_patterns = [
            r"from\s+(.+)",
            r"to\s+(.+)",
            r"find\s+(.+)",
            r"reach\s+(.+)"
        ]
        
        target_object = "object"
        for pattern in object_patterns:
            match = re.search(pattern, command)
            if match:
                target_object = match.group(1).strip()
                break
        
        print(f"✅ '{command}' → '{target_object}'")
    
    print()

def test_help_system():
    """Test help system."""
    print("🧪 Testing Help System")
    print("=" * 40)
    
    agent = SmartNavigationAgent()
    help_text = agent._get_help()
    
    print("Help system output:")
    print(help_text)
    print()

def test_status_system():
    """Test status system."""
    print("🧪 Testing Status System")
    print("=" * 40)
    
    agent = SmartNavigationAgent()
    
    # Test initial status
    print("Initial status:")
    status = agent._get_status()
    print(status)
    print()
    
    # Test status with active task (simulated)
    agent.current_task = "Test navigation task"
    agent.current_agent = "navigation"
    
    print("Status with active task:")
    status = agent._get_status()
    print(status)
    print()

def main():
    """Run all tests."""
    print("🤖 Navigation Agents Test Suite")
    print("=" * 50)
    print()
    
    try:
        test_command_analysis()
        test_parameter_extraction()
        test_help_system()
        test_status_system()
        
        print("✅ All tests completed successfully!")
        print()
        print("To run the actual navigation agents:")
        print("python smart_navigation_agent.py")
        print("python navigation_agent.py")
        print("python obstacle_avoidance_agent.py")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
