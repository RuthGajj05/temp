#!/usr/bin/env python3
"""
Example script demonstrating the navigation agents
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from smart_navigation_agent import SmartNavigationAgent

def demo_navigation_agents():
    """Demonstrate the navigation agents with example commands."""
    print("🤖 PicarX Navigation Agents Demo")
    print("=" * 50)
    
    agent = SmartNavigationAgent()
    
    # Example commands
    example_commands = [
        "navigate until 30cm from the couch",
        "drive to 20cm from the wall", 
        "navigate to the red ball",
        "find the exit while avoiding obstacles",
        "status",
        "stop"
    ]
    
    print("Example commands that will be demonstrated:")
    for i, cmd in enumerate(example_commands, 1):
        print(f"{i}. '{cmd}'")
    print()
    
    # Simulate the commands (without actually running them)
    print("🚀 Simulating navigation commands...")
    print()
    
    for i, command in enumerate(example_commands, 1):
        print(f"--- Example {i}: '{command}' ---")
        
        # Analyze the command without executing
        task_type = agent._analyze_command(command.lower())
        print(f"Detected task type: {task_type}")
        
        if task_type == "distance_navigation":
            print("→ Would use Distance Navigation Agent")
            print("→ Would extract distance and target object")
            print("→ Would navigate until specified distance")
        elif task_type == "obstacle_avoidance":
            print("→ Would use Obstacle Avoidance Agent")
            print("→ Would navigate while avoiding obstacles")
            print("→ Would use camera and sensors for guidance")
        elif task_type == "status":
            print("→ Would check current navigation status")
        elif task_type == "stop":
            print("→ Would stop all navigation tasks")
        
        print()

def interactive_demo():
    """Run an interactive demo."""
    print("🤖 Interactive Navigation Demo")
    print("=" * 40)
    print("This demo will show you how the agents work.")
    print("Commands will be analyzed but not executed.")
    print("Type 'quit' to exit.")
    print()
    
    agent = SmartNavigationAgent()
    
    while True:
        try:
            command = input("Demo> ").strip()
            
            if command.lower() in ["quit", "exit"]:
                print("👋 Demo finished!")
                break
            elif command.lower() == "help":
                print(agent._get_help())
            else:
                # Analyze the command
                task_type = agent._analyze_command(command.lower())
                print(f"📊 Analysis: {task_type}")
                
                if task_type == "distance_navigation":
                    print("🎯 This would use Distance Navigation Agent")
                    print("   - Extracts distance and target object")
                    print("   - Navigates until specified distance")
                    print("   - Uses ultrasonic sensor for distance measurement")
                elif task_type == "obstacle_avoidance":
                    print("🚧 This would use Obstacle Avoidance Agent")
                    print("   - Navigates while avoiding obstacles")
                    print("   - Uses camera and sensors for guidance")
                    print("   - Implements multiple avoidance strategies")
                elif task_type == "status":
                    print("📊 This would check navigation status")
                elif task_type == "stop":
                    print("⏹️ This would stop all navigation tasks")
                else:
                    print("❓ Unknown command type")
                
                print()
                
        except KeyboardInterrupt:
            print("\n👋 Demo finished!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("Choose demo mode:")
    print("1. Show example commands")
    print("2. Interactive analysis demo")
    print("3. Run actual navigation agents (requires robot)")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        demo_navigation_agents()
    elif choice == "2":
        interactive_demo()
    elif choice == "3":
        print("🚀 Starting actual navigation agents...")
        from smart_navigation_agent import main
        main()
    else:
        print("❌ Invalid choice. Running example commands demo.")
        demo_navigation_agents()
