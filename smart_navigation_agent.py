#!/usr/bin/env python3

import sys
import os
import time
import re
from typing import Optional, Dict, Any

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from primitives import *
from openai import OpenAI
from keys import OPENAI_API_KEY
from navigation_agent import NavigationAgent
from obstacle_avoidance_agent import ObstacleAvoidanceAgent

class SmartNavigationAgent:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.navigation_agent = NavigationAgent()
        self.obstacle_agent = ObstacleAvoidanceAgent()
        self.current_task = None
        self.current_agent = None
        
    def process_command(self, command: str) -> str:
        try:
            command_lower = command.lower().strip()
            
            task_type = self._analyze_command(command_lower)
            
            if task_type == "distance_navigation":
                return self._handle_distance_navigation(command_lower)
            elif task_type == "obstacle_avoidance":
                return self._handle_obstacle_avoidance(command_lower)
            elif task_type == "status":
                return self._get_status()
            elif task_type == "stop":
                return self._handle_stop()
            else:
                return f"❌ Unknown command type: {task_type}"
                
        except Exception as e:
            return f"❌ Error processing command: {str(e)}"

    def _analyze_command(self, command: str) -> str:
        distance_patterns = [
            r"navigate.*?(\d+(?:\.\d+)?)\s*cm",
            r"drive.*?(\d+(?:\.\d+)?)\s*cm",
            r"go.*?(\d+(?:\.\d+)?)\s*cm",
            r"move.*?(\d+(?:\.\d+)?)\s*cm",
            r"until.*?(\d+(?:\.\d+)?)\s*cm",
            r"to.*?(\d+(?:\.\d+)?)\s*cm",
            r"nearest.*?object",
            r"closest.*?object"
        ]
        
        obstacle_patterns = [
            r"avoid.*?obstacle",
            r"around.*?obstacle",
            r"navigate.*?obstacle",
            r"drive.*?obstacle",
            r"go.*?obstacle",
            r"move.*?obstacle",
            r"while.*?avoiding",
            r"obstacle.*?avoidance"
        ]
        
        status_patterns = [
            r"status",
            r"state",
            r"info",
            r"help"
        ]
        
        stop_patterns = [
            r"stop",
            r"halt",
            r"quit",
            r"exit"
        ]
        
        for pattern in distance_patterns:
            if re.search(pattern, command):
                return "distance_navigation"
        
        for pattern in obstacle_patterns:
            if re.search(pattern, command):
                return "obstacle_avoidance"
        
        for pattern in status_patterns:
            if re.search(pattern, command):
                return "status"
        
        for pattern in stop_patterns:
            if re.search(pattern, command):
                return "stop"
        
        return "unknown"

    def _handle_distance_navigation(self, command: str) -> str:
        try:
            distance_match = re.search(r'(\d+(?:\.\d+)?)\s*cm', command)
            if distance_match:
                distance = float(distance_match.group(1))
            else:
                distance = 30.0
            
            nearest_match = re.search(r'(nearest|closest)\s+object', command)
            if nearest_match:
                target_object = "nearest object"
            else:
                object_patterns = [
                    r'from\s+the\s+([^,\s]+(?:\s+[^,\s]+)*)',
                    r'to\s+the\s+([^,\s]+(?:\s+[^,\s]+)*)',
                    r'near\s+the\s+([^,\s]+(?:\s+[^,\s]+)*)',
                    r'([^,\s]+(?:\s+[^,\s]+)*)\s+until',
                    r'([^,\s]+(?:\s+[^,\s]+)*)\s+at',
                    r'([^,\s]+(?:\s+[^,\s]+)*)\s+to'
                ]
                
                target_object = "object"
                for pattern in object_patterns:
                    match = re.search(pattern, command)
                    if match:
                        target_object = match.group(1).strip()
                        break
            
            print(f"🎯 Distance Navigation: {target_object} at {distance}cm")
            
            self.current_task = "distance_navigation"
            self.current_agent = self.navigation_agent
            
            result = self.navigation_agent.navigate_to_distance(distance, target_object)
            
            self.current_task = None
            self.current_agent = None
            
            return result
            
        except Exception as e:
            return f"❌ Distance navigation error: {str(e)}"

    def _handle_obstacle_avoidance(self, command: str) -> str:
        try:
            object_patterns = [
                r'to\s+the\s+([^,\s]+(?:\s+[^,\s]+)*)',
                r'near\s+the\s+([^,\s]+(?:\s+[^,\s]+)*)',
                r'([^,\s]+(?:\s+[^,\s]+)*)\s+while',
                r'([^,\s]+(?:\s+[^,\s]+)*)\s+avoiding',
                r'([^,\s]+(?:\s+[^,\s]+)*)\s+around'
            ]
            
            target_object = "target"
            for pattern in object_patterns:
                match = re.search(pattern, command)
                if match:
                    target_object = match.group(1).strip()
                    break
            
            print(f"🎯 Obstacle Avoidance: {target_object}")
            
            self.current_task = "obstacle_avoidance"
            self.current_agent = self.obstacle_agent
            
            result = self.obstacle_agent.navigate_around_obstacles(target_object)
            
            self.current_task = None
            self.current_agent = None
            
            return result
            
        except Exception as e:
            return f"❌ Obstacle avoidance error: {str(e)}"

    def _get_status(self) -> str:
        nav_status = self.navigation_agent.get_status()
        obstacle_status = self.obstacle_agent.get_status()
        
        return f"""
Smart Navigation Agent Status:
Current Task: {self.current_task}
Current Agent: {self.current_agent.__class__.__name__ if self.current_agent else None}

Navigation Agent:
{nav_status}

Obstacle Avoidance Agent:
{obstacle_status}
"""

    def _handle_stop(self) -> str:
        try:
            stop_result = stop()
            self.current_task = None
            self.current_agent = None
            return f"✅ All agents stopped. {stop_result}"
        except Exception as e:
            return f"❌ Stop error: {str(e)}"

def main():
    agent = SmartNavigationAgent()
    
    print("🤖 Smart Navigation Agent Ready!")
    print("Natural language commands supported:")
    print("  • 'navigate until 30cm from the couch'")
    print("  • 'drive to the nearest object until 20cm away'")
    print("  • 'navigate all obstacles until you reach the red ball'")
    print("  • 'status' - Get agent status")
    print("  • 'stop' - Stop all movement")
    print("  • 'quit' - Exit")
    
    while True:
        try:
            command = input("\nSmartNav> ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif command.lower() == 'stop':
                result = agent._handle_stop()
                print(result)
            else:
                result = agent.process_command(command)
                print(f"Result: {result}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()