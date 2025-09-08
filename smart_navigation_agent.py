#!/usr/bin/env python3
"""
Smart Navigation Agent for PicarX Robot
Orchestrates navigation and obstacle avoidance agents based on user prompts
"""

import sys
import os
import time
import re
from typing import Optional, Dict, Any

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from primitives import *
from openai import OpenAI
from keys import OPENAI_API_KEY
from navigation_agent import NavigationAgent
from obstacle_avoidance_agent import ObstacleAvoidanceAgent

class SmartNavigationAgent:
    def __init__(self):
        """Initialize the Smart Navigation Agent."""
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.navigation_agent = NavigationAgent()
        self.obstacle_agent = ObstacleAvoidanceAgent()
        self.current_task = None
        self.current_agent = None
        
    def process_command(self, command: str) -> str:
        """
        Process a natural language navigation command.
        
        Args:
            command: Natural language command like "navigate until 30cm from the couch"
        """
        try:
            command_lower = command.lower().strip()
            
            # Analyze the command to determine which agent to use
            task_type = self._analyze_command(command_lower)
            
            if task_type == "distance_navigation":
                return self._handle_distance_navigation(command_lower)
            elif task_type == "obstacle_avoidance":
                return self._handle_obstacle_avoidance(command_lower)
            elif task_type == "status":
                return self._get_status()
            elif task_type == "stop":
                return self._stop_all()
            elif task_type == "help":
                return self._get_help()
            else:
                return f"❌ I don't understand that command. {self._get_help()}"
                
        except Exception as e:
            return f"❌ Error processing command: {str(e)}"
    
    def _analyze_command(self, command: str) -> str:
        """Analyze the command to determine the appropriate agent."""
        
        # Check for distance-based navigation patterns
        distance_patterns = [
            r"navigate.*?(\d+)\s*cm",
            r"drive.*?(\d+)\s*cm",
            r"go.*?(\d+)\s*cm",
            r"move.*?(\d+)\s*cm",
            r"until.*?(\d+)\s*cm",
            r"(\d+)\s*cm.*?from",
            r"(\d+)\s*centimeter"
        ]
        
        for pattern in distance_patterns:
            if re.search(pattern, command):
                return "distance_navigation"
        
        # Check for obstacle avoidance patterns
        obstacle_patterns = [
            r"avoid.*?obstacle",
            r"navigate.*?around",
            r"find.*?path",
            r"reach.*?destination",
            r"get.*?to.*?without",
            r"obstacle.*?avoidance",
            r"navigate.*?all.*?obstacle",
            r"around.*?obstacle"
        ]
        
        for pattern in obstacle_patterns:
            if re.search(pattern, command):
                return "obstacle_avoidance"
        
        # Check for status/control commands
        if any(word in command for word in ["status", "check", "current"]):
            return "status"
        elif any(word in command for word in ["stop", "halt", "quit", "exit"]):
            return "stop"
        elif any(word in command for word in ["help", "commands", "what can you do"]):
            return "help"
        
        # Default to obstacle avoidance for complex navigation
        if any(word in command for word in ["navigate", "drive", "go", "move", "find", "reach"]):
            return "obstacle_avoidance"
        
        return "unknown"
    
    def _handle_distance_navigation(self, command: str) -> str:
        """Handle distance-based navigation commands."""
        try:
            # Extract distance and object from command
            distance_match = re.search(r"(\d+)\s*cm", command)
            if not distance_match:
                return "❌ Please specify a distance in cm (e.g., 'navigate until 30cm from the couch')"
            
            distance = float(distance_match.group(1))
            
            # Extract target object
            object_patterns = [
                r"from\s+(.+)",
                r"to\s+(.+)",
                r"near\s+(.+)",
                r"close\s+to\s+(.+)"
            ]
            
            target_object = "object"
            for pattern in object_patterns:
                match = re.search(pattern, command)
                if match:
                    target_object = match.group(1).strip()
                    break
            
            # Set current task
            self.current_task = f"Navigate to {distance}cm from {target_object}"
            self.current_agent = "navigation"
            
            print(f"🎯 Task: {self.current_task}")
            print(f"🤖 Using: Distance Navigation Agent")
            
            # Execute navigation
            result = self.navigation_agent.navigate_to_distance(distance, target_object)
            
            # Clear current task
            self.current_task = None
            self.current_agent = None
            
            return result
            
        except Exception as e:
            self.current_task = None
            self.current_agent = None
            return f"❌ Distance navigation error: {str(e)}"
    
    def _handle_obstacle_avoidance(self, command: str) -> str:
        """Handle obstacle avoidance navigation commands."""
        try:
            # Extract target object
            target_object = "destination"
            
            # Look for target objects in the command
            object_patterns = [
                r"to\s+(.+)",
                r"find\s+(.+)",
                r"reach\s+(.+)",
                r"get\s+to\s+(.+)",
                r"navigate\s+to\s+(.+)",
                r"drive\s+to\s+(.+)"
            ]
            
            for pattern in object_patterns:
                match = re.search(pattern, command)
                if match:
                    target_object = match.group(1).strip()
                    break
            
            # Set current task
            self.current_task = f"Navigate to {target_object} while avoiding obstacles"
            self.current_agent = "obstacle_avoidance"
            
            print(f"🎯 Task: {self.current_task}")
            print(f"🤖 Using: Obstacle Avoidance Agent")
            
            # Execute navigation
            result = self.obstacle_agent.navigate_around_obstacles(target_object)
            
            # Clear current task
            self.current_task = None
            self.current_agent = None
            
            return result
            
        except Exception as e:
            self.current_task = None
            self.current_agent = None
            return f"❌ Obstacle avoidance error: {str(e)}"
    
    def _get_status(self) -> str:
        """Get current status of all agents."""
        status_parts = []
        
        if self.current_task:
            status_parts.append(f"Current Task: {self.current_task}")
            status_parts.append(f"Active Agent: {self.current_agent}")
        
        # Get status from active agent
        if self.current_agent == "navigation":
            agent_status = self.navigation_agent.get_navigation_status()
        elif self.current_agent == "obstacle_avoidance":
            agent_status = self.obstacle_agent.get_navigation_status()
        else:
            agent_status = "No active navigation task"
        
        status_parts.append(f"Agent Status: {agent_status}")
        
        # Get robot state
        try:
            robot_status = get_robot_status()
            status_parts.append(f"Robot State: {robot_status}")
        except:
            status_parts.append("Robot State: Unable to read")
        
        return "\n".join(status_parts)
    
    def _stop_all(self) -> str:
        """Stop all navigation tasks."""
        results = []
        
        # Stop navigation agent
        nav_result = self.navigation_agent.stop_navigation()
        results.append(f"Navigation Agent: {nav_result}")
        
        # Stop obstacle avoidance agent
        obs_result = self.obstacle_agent.stop_navigation()
        results.append(f"Obstacle Avoidance Agent: {obs_result}")
        
        # Clear current task
        self.current_task = None
        self.current_agent = None
        
        return "\n".join(results)
    
    def _get_help(self) -> str:
        """Get help information."""
        return """
🤖 Smart Navigation Agent Help

📋 Available Commands:

🎯 Distance Navigation:
• "navigate until 30cm from the couch"
• "drive to 20cm from the wall"
• "go until 15cm from the table"
• "move to 25cm from the chair"

🚧 Obstacle Avoidance:
• "navigate to the red ball"
• "find the exit"
• "reach the kitchen"
• "navigate around all obstacles to the door"
• "drive to the couch while avoiding obstacles"

📊 Status & Control:
• "status" - Check current navigation status
• "stop" - Stop all navigation tasks
• "help" - Show this help message
• "quit" - Exit the program

💡 Tips:
• Be specific about distances (use "cm")
• Mention target objects clearly
• Use "avoid obstacles" for complex navigation
• Check status regularly during long tasks
        """

def main():
    """Interactive smart navigation agent."""
    print("🤖 PicarX Smart Navigation Agent")
    print("=" * 45)
    print("I can help you navigate your robot in two ways:")
    print("1. 🎯 Distance Navigation: 'navigate until 30cm from the couch'")
    print("2. 🚧 Obstacle Avoidance: 'navigate to the red ball while avoiding obstacles'")
    print()
    print("Type 'help' for more commands or 'quit' to exit.")
    print()
    
    agent = SmartNavigationAgent()
    
    while True:
        try:
            command = input("SmartNav> ").strip()
            
            if command.lower() in ["quit", "exit"]:
                print("👋 Goodbye!")
                break
            elif command.lower() == "help":
                print(agent._get_help())
            else:
                result = agent.process_command(command)
                print(f"\n{result}\n")
                
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
            agent._stop_all()
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
