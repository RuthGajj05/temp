# PicarX Smart Navigation System

A comprehensive navigation system for the PicarX robot with specialized agents for different navigation tasks.

## 🎯 Overview

This system provides three specialized agents for different navigation scenarios:

1. **Distance Navigation Agent** - Navigate to a specific distance from objects
2. **Obstacle Avoidance Agent** - Navigate while avoiding obstacles using sensors and camera
3. **Smart Navigation Agent** - Orchestrates the other agents based on natural language commands

## 🚀 Quick Start

### Basic Usage

```bash
# Run the smart navigation agent (recommended)
python smart_navigation_agent.py

# Or run individual agents
python navigation_agent.py
python obstacle_avoidance_agent.py
```

### Example Commands

```bash
# Distance-based navigation
"navigate until 30cm from the couch"
"drive to 20cm from the wall"
"go until 15cm from the table"

# Obstacle avoidance navigation
"navigate to the red ball"
"find the exit while avoiding obstacles"
"reach the kitchen"
"navigate around all obstacles to the door"
```

## 🤖 Agent Descriptions

### 1. Distance Navigation Agent

**Purpose**: Navigate to a specific distance from the nearest object.

**Key Features**:
- Uses ultrasonic sensor for distance measurement
- Continuous distance monitoring
- Safety stops at close distances
- Obstacle detection during movement

**Best For**:
- "Navigate until 30cm from the couch"
- "Drive to 20cm from the wall"
- Precise distance-based positioning

**How It Works**:
1. Resets robot to starting position
2. Continuously reads ultrasonic sensor
3. Moves forward with obstacle checking
4. Stops when target distance is reached
5. Provides safety stops for close distances

### 2. Obstacle Avoidance Agent

**Purpose**: Navigate to a target while intelligently avoiding obstacles.

**Key Features**:
- Multi-sensor navigation (ultrasonic, grayscale, camera)
- Multiple avoidance strategies
- Visual navigation with camera
- Adaptive movement patterns
- Maximum attempt limits for safety

**Best For**:
- "Navigate to the red ball"
- "Find the exit"
- "Reach the kitchen"
- Complex environments with obstacles

**How It Works**:
1. Initializes camera for visual navigation
2. Takes photos for analysis
3. Uses multiple sensors for obstacle detection
4. Implements various avoidance strategies:
   - Turn left and try forward
   - Turn right and try forward
   - Back up and try different angle
5. Adapts movement speed based on distance to target

### 3. Smart Navigation Agent

**Purpose**: Orchestrates the other agents based on natural language commands.

**Key Features**:
- Natural language command processing
- Automatic agent selection
- Command analysis and parsing
- Status monitoring and control
- Help system

**Best For**:
- Natural language interaction
- Complex command interpretation
- Multi-agent coordination

**How It Works**:
1. Analyzes user commands using regex patterns
2. Determines appropriate agent based on command type
3. Extracts parameters (distance, target objects)
4. Delegates to appropriate specialized agent
5. Provides status and control functions

## 🛠️ Technical Details

### Prerequisites

```bash
pip install openai
```

### Required Files

- `primitives.py` - Robot control functions
- `keys.py` - OpenAI API key configuration
- `navigation_agent.py` - Distance navigation
- `obstacle_avoidance_agent.py` - Obstacle avoidance
- `smart_navigation_agent.py` - Main orchestrator

### Command Analysis

The Smart Navigation Agent uses regex patterns to analyze commands:

**Distance Navigation Patterns**:
```python
r"navigate.*?(\d+)\s*cm"
r"drive.*?(\d+)\s*cm"
r"until.*?(\d+)\s*cm"
r"(\d+)\s*cm.*?from"
```

**Obstacle Avoidance Patterns**:
```python
r"avoid.*?obstacle"
r"navigate.*?around"
r"find.*?path"
r"reach.*?destination"
```

### Safety Features

- **Speed Limits**: All movement capped at 0-100 range
- **Distance Safety**: Automatic stops at <5cm
- **Obstacle Detection**: Real-time obstacle checking
- **Timeout Protection**: Maximum attempt limits
- **Emergency Stop**: Immediate stop functionality

## 📊 Usage Examples

### Example 1: Distance Navigation

```python
from smart_navigation_agent import SmartNavigationAgent

agent = SmartNavigationAgent()

# Navigate to 30cm from couch
result = agent.process_command("navigate until 30cm from the couch")
print(result)
# Output: "✅ Navigation complete! Reached 30cm from couch. Final distance: 29.8cm."
```

### Example 2: Obstacle Avoidance

```python
# Navigate to red ball while avoiding obstacles
result = agent.process_command("navigate to the red ball")
print(result)
# Output: "✅ Successfully reached red ball! Final distance: 18.5cm."
```

### Example 3: Status Monitoring

```python
# Check current status
status = agent.process_command("status")
print(status)
# Output: Current task, active agent, robot state, etc.
```

## 🎮 Interactive Mode

### Smart Navigation Agent

```bash
python smart_navigation_agent.py
```

**Available Commands**:
- `navigate until 30cm from the couch`
- `drive to 20cm from the wall`
- `navigate to the red ball`
- `find the exit while avoiding obstacles`
- `status` - Check current status
- `stop` - Stop all navigation
- `help` - Show help information
- `quit` - Exit program

### Individual Agents

```bash
# Distance Navigation Agent
python navigation_agent.py

# Obstacle Avoidance Agent  
python obstacle_avoidance_agent.py
```

## 🔧 Configuration

### API Key Setup

Edit `keys.py`:
```python
OPENAI_API_KEY = "your-openai-api-key-here"
```

### Customization

**Navigation Parameters**:
```python
# In navigation_agent.py
target_distance = 30.0  # cm
safety_distance = 5.0   # cm

# In obstacle_avoidance_agent.py
max_attempts = 50       # Maximum navigation attempts
movement_speed = 30     # Default movement speed
```

**Camera Settings**:
```python
# Camera initialization
init_camera()  # Uses Vilib camera system
```

## 🚨 Troubleshooting

### Common Issues

1. **API Key Not Found**
   ```
   Error: OPENAI_API_KEY not found in keys.py
   ```
   Solution: Add your OpenAI API key to `keys.py`

2. **Camera Initialization Failed**
   ```
   Camera initialization timeout - check camera connection
   ```
   Solution: Ensure camera is properly connected and Vilib is installed

3. **Navigation Stuck**
   ```
   Navigation timeout after 50 attempts
   ```
   Solution: Check for obstacles, try different approach, or reset robot

4. **Distance Reading Errors**
   ```
   Error reading distance: [sensor error]
   ```
   Solution: Check ultrasonic sensor connection

### Debug Mode

Enable verbose logging by modifying the agents:
```python
# Add debug prints
print(f"Debug: Current distance: {current_distance}")
print(f"Debug: Movement strategy: {strategy}")
```

## 📈 Performance Tips

1. **Good Lighting**: Ensure adequate lighting for camera-based navigation
2. **Clear Commands**: Use specific, clear commands for better results
3. **Regular Status Checks**: Monitor progress during long navigation tasks
4. **Obstacle Management**: Clear major obstacles before complex navigation
5. **Speed Adjustment**: Use appropriate speeds for the environment

## 🔮 Future Enhancements

- **Voice Integration**: Add speech recognition and TTS
- **Advanced Vision**: Object recognition and tracking
- **Path Planning**: A* or other pathfinding algorithms
- **Multi-Robot**: Coordinate multiple robots
- **Learning**: Improve navigation based on experience
- **Web Interface**: Browser-based control panel

## 📚 Related Files

- `primitives.py` - Core robot control functions
- `navigation_agent.py` - Distance-based navigation
- `obstacle_avoidance_agent.py` - Obstacle avoidance navigation
- `smart_navigation_agent.py` - Main orchestrator agent
- `navigation_example.py` - Usage examples
- `NAVIGATION_README.md` - This documentation

## 🎯 Best Practices

1. **Start Simple**: Begin with distance navigation before complex tasks
2. **Monitor Progress**: Use status commands regularly
3. **Clear Environment**: Remove unnecessary obstacles when possible
4. **Appropriate Speeds**: Use slower speeds in cluttered environments
5. **Safety First**: Always have a stop command ready
6. **Test Commands**: Use the demo mode to test command parsing
7. **Backup Plans**: Have manual control ready for complex situations

This navigation system provides a powerful and flexible way to control your PicarX robot for various navigation tasks!
