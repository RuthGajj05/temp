"""
PicarX Robot Primitives - Production Ready Implementation
Addresses all requirements from the Final Project Prompt
"""

from picarx import Picarx
import time
import json
from typing import List, Optional, Union
from robot_hat import Music
import os
import threading

# Global variable to store the singleton PicarX instance
_picarx = None

# Dictionary to track current angles of all servo motors
_servo_angles = {
    'dir_servo': 0,    # Direction servo angle (steering)
    'cam_pan': 0,      # Camera pan servo angle (left/right)
    'cam_tilt': 0      # Camera tilt servo angle (up/down)
}

# Flag to track if the camera system has been initialized
_vilib_initialized = False

# Flag for continuous movement control
_continuous_movement = False
_movement_thread = None

def get_picarx() -> Picarx:
    """Get the singleton PicarX instance."""
    global _picarx
    if _picarx is None:
        _picarx = Picarx()
    return _picarx

def clamp_value(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))

def check_obstacle_safe(distance_threshold: float = 15.0) -> bool:
    """Check if path is clear of obstacles."""
    try:
        px = get_picarx()
        distance = px.ultrasonic.read()
        return distance > distance_threshold
    except:
        return True  # If sensor fails, assume safe

def function_tool(func):
    """Decorator to mark functions as callable tools for agents."""
    func._is_tool = True
    return func

@function_tool
def reset() -> str:
    """Reset all servos to center position and stop motors."""
    global _servo_angles, _continuous_movement
    try:
        px = get_picarx()
        px.set_dir_servo_angle(0)
        px.set_cam_pan_angle(0)
        px.set_cam_tilt_angle(0)
        px.stop()
        _servo_angles['dir_servo'] = 0
        _servo_angles['cam_pan'] = 0
        _servo_angles['cam_tilt'] = 0
        _continuous_movement = False
        return "Robot reset: All servos centered, motors stopped"
    except Exception as e:
        return f"Reset failed: {str(e)}"

@function_tool
def move_forward(speed: int, duration: Optional[float] = None, check_obstacles: bool = True) -> str:
    """Move forward at specified speed for given duration or continuously until stop()."""
    global _servo_angles, _continuous_movement
    
    # Validate and clamp speed
    speed = int(clamp_value(speed, 0, 100))
    
    try:
        px = get_picarx()
        
        if duration is None:
            # Continuous movement mode
            _continuous_movement = True
            px.forward(speed)
            _servo_angles['dir_servo'] = 0
            return f"Moving forward continuously at {speed}% speed. Call stop() to halt."
        else:
            # Timed movement with optional obstacle checking
            px.forward(speed)
            _servo_angles['dir_servo'] = 0
            
            if check_obstacles:
                # Check for obstacles every 50ms during movement
                start_time = time.time()
                while time.time() - start_time < duration:
                    if not check_obstacle_safe():
                        px.stop()
                        return f"Movement stopped early: Obstacle detected at {px.ultrasonic.read():.1f}cm"
                    time.sleep(0.05)
            else:
                time.sleep(duration)
            
            px.stop()
            return f"Moved forward for {duration:.2f}s at {speed}% speed"
            
    except Exception as e:
        return f"Move forward failed: {str(e)}"

@function_tool
def move_backward(speed: int, duration: Optional[float] = None) -> str:
    """Move backward at specified speed for given duration or continuously until stop()."""
    global _servo_angles, _continuous_movement
    
    # Validate and clamp speed
    speed = int(clamp_value(speed, 0, 100))
    
    try:
        px = get_picarx()
        
        if duration is None:
            # Continuous movement mode
            _continuous_movement = True
            px.backward(speed)
            _servo_angles['dir_servo'] = 0
            return f"Moving backward continuously at {speed}% speed. Call stop() to halt."
        else:
            # Timed movement
            px.backward(speed)
            _servo_angles['dir_servo'] = 0
            time.sleep(duration)
            px.stop()
            return f"Moved backward for {duration:.2f}s at {speed}% speed"
            
    except Exception as e:
        return f"Move backward failed: {str(e)}"

@function_tool
def turn_left(angle: float, speed: int, duration: Optional[float] = None) -> str:
    """Turn left by steering at specified angle while moving."""
    global _servo_angles, _continuous_movement
    
    # Validate and clamp parameters
    angle = clamp_value(angle, -30, 30)
    speed = int(clamp_value(speed, 0, 100))
    
    try:
        px = get_picarx()
        
        # Set steering angle for left turn
        px.set_dir_servo_angle(-angle)
        _servo_angles['dir_servo'] = -angle
        
        if duration is None:
            # Continuous movement mode
            _continuous_movement = True
            px.forward(speed)
            return f"Turning left continuously at {angle}° angle, {speed}% speed. Call stop() to halt."
        else:
            # Timed movement
            px.forward(speed)
            time.sleep(duration)
            px.stop()
            # Reset steering to straight
            px.set_dir_servo_angle(0)
            _servo_angles['dir_servo'] = 0
            return f"Turned left at {angle}° angle for {duration:.2f}s at {speed}% speed"
            
    except Exception as e:
        return f"Turn left failed: {str(e)}"

@function_tool
def turn_right(angle: float, speed: int, duration: Optional[float] = None) -> str:
    """Turn right by steering at specified angle while moving."""
    global _servo_angles, _continuous_movement
    
    # Validate and clamp parameters
    angle = clamp_value(angle, -30, 30)
    speed = int(clamp_value(speed, 0, 100))
    
    try:
        px = get_picarx()
        
        # Set steering angle for right turn
        px.set_dir_servo_angle(angle)
        _servo_angles['dir_servo'] = angle
        
        if duration is None:
            # Continuous movement mode
            _continuous_movement = True
            px.forward(speed)
            return f"Turning right continuously at {angle}° angle, {speed}% speed. Call stop() to halt."
        else:
            # Timed movement
            px.forward(speed)
            time.sleep(duration)
            px.stop()
            # Reset steering to straight
            px.set_dir_servo_angle(0)
            _servo_angles['dir_servo'] = 0
            return f"Turned right at {angle}° angle for {duration:.2f}s at {speed}% speed"
            
    except Exception as e:
        return f"Turn right failed: {str(e)}"

@function_tool
def turn_in_place_left(angle: float, speed: int, duration: Optional[float] = None) -> str:
    """Turn left in place using differential motor control (no forward movement)."""
    global _servo_angles, _continuous_movement
    
    # Validate and clamp parameters
    angle = clamp_value(angle, 0, 360)  # Allow full rotation angles
    speed = int(clamp_value(speed, 0, 100))
    
    try:
        px = get_picarx()
        
        # Calculate duration based on angle if not provided
        if duration is None:
            # Estimate duration based on angle (roughly 1 second per 90 degrees at 50% speed)
            duration = (angle / 90.0) * (50.0 / max(speed, 1))  # Scale by speed
            duration = max(0.5, min(duration, 10.0))  # Clamp between 0.5 and 10 seconds
        
        # Set motor calibration for left turn in place: left motor backward, right motor forward
        px.cali_dir_value = [-1, 1]
        _servo_angles['dir_servo'] = 0  # Keep steering straight
        
        if _continuous_movement:
            # Continuous movement mode
            px.forward(speed)
            return f"Turning left in place continuously at {speed}% speed. Call stop() to halt."
        else:
            # Timed movement
            px.forward(speed)
            time.sleep(duration)
            px.stop()
            # Reset motor calibration to normal
            px.cali_dir_value = [1, 1]
            return f"Turned left in place by ~{angle}° for {duration:.2f}s at {speed}% speed"
            
    except Exception as e:
        # Ensure we reset calibration even on error
        try:
            px = get_picarx()
            px.cali_dir_value = [1, 1]
        except:
            pass
        return f"Turn in place left failed: {str(e)}"

@function_tool
def turn_in_place_right(angle: float, speed: int, duration: Optional[float] = None) -> str:
    """Turn right in place using differential motor control (no forward movement)."""
    global _servo_angles, _continuous_movement
    
    # Validate and clamp parameters
    angle = clamp_value(angle, 0, 360)  # Allow full rotation angles
    speed = int(clamp_value(speed, 0, 100))
    
    try:
        px = get_picarx()
        
        # Calculate duration based on angle if not provided
        if duration is None:
            # Estimate duration based on angle (roughly 1 second per 90 degrees at 50% speed)
            duration = (angle / 90.0) * (50.0 / max(speed, 1))  # Scale by speed
            duration = max(0.5, min(duration, 10.0))  # Clamp between 0.5 and 10 seconds
        
        # Set motor calibration for right turn in place: left motor forward, right motor backward
        px.cali_dir_value = [1, -1]
        _servo_angles['dir_servo'] = 0  # Keep steering straight
        
        if _continuous_movement:
            # Continuous movement mode
            px.forward(speed)
            return f"Turning right in place continuously at {speed}% speed. Call stop() to halt."
        else:
            # Timed movement
            px.forward(speed)
            time.sleep(duration)
            px.stop()
            # Reset motor calibration to normal
            px.cali_dir_value = [1, 1]
            return f"Turned right in place by ~{angle}° for {duration:.2f}s at {speed}% speed"
            
    except Exception as e:
        # Ensure we reset calibration even on error
        try:
            px = get_picarx()
            px.cali_dir_value = [1, 1]
        except:
            pass
        return f"Turn in place right failed: {str(e)}"

@function_tool
def stop() -> str:
    """Stop all robot movement immediately."""
    global _continuous_movement
    try:
        px = get_picarx()
        px.stop()
        _continuous_movement = False
        return "Robot stopped"
    except Exception as e:
        return f"Stop failed: {str(e)}"

@function_tool
def set_motor_speed(motor_id: int, speed: int) -> str:
    """Set the speed of an individual motor."""
    # Validate parameters
    motor_id = int(clamp_value(motor_id, 1, 2))
    speed = int(clamp_value(speed, -100, 100))
    
    try:
        px = get_picarx()
        px.set_motor_speed(motor_id, speed)
        motor_name = "left" if motor_id == 1 else "right"
        return f"Set {motor_name} motor speed to {speed}%"
    except Exception as e:
        return f"Set motor speed failed: {str(e)}"

@function_tool
def set_dir_servo_angle(angle: float) -> str:
    """Set the direction servo angle for steering."""
    # Validate and clamp angle
    angle = clamp_value(angle, -30, 30)
    
    try:
        global _servo_angles
        px = get_picarx()
        px.set_dir_servo_angle(angle)
        _servo_angles['dir_servo'] = angle
        return f"Direction servo set to {angle}°"
    except Exception as e:
        return f"Set direction servo failed: {str(e)}"

@function_tool
def set_cam_pan_angle(angle: float) -> str:
    """Set the camera pan servo angle (left/right)."""
    # Validate and clamp angle
    angle = clamp_value(angle, -35, 35)
    
    try:
        global _servo_angles
        px = get_picarx()
        px.set_cam_pan_angle(angle)
        _servo_angles['cam_pan'] = angle
        return f"Camera pan servo set to {angle}°"
    except Exception as e:
        return f"Set camera pan failed: {str(e)}"

@function_tool
def set_cam_tilt_angle(angle: float) -> str:
    """Set the camera tilt servo angle (up/down)."""
    # Validate and clamp angle
    angle = clamp_value(angle, -35, 35)
    
    try:
        global _servo_angles
        px = get_picarx()
        px.set_cam_tilt_angle(angle)
        _servo_angles['cam_tilt'] = angle
        return f"Camera tilt servo set to {angle}°"
    except Exception as e:
        return f"Set camera tilt failed: {str(e)}"

@function_tool
def get_servo_angles() -> str:
    """Get current angles of all servos."""
    global _servo_angles
    return json.dumps(_servo_angles, indent=2)

@function_tool
def get_dir_servo_angle() -> str:
    """Get current steering servo angle."""
    global _servo_angles
    return f"Direction servo angle: {_servo_angles['dir_servo']}°"

@function_tool
def get_cam_pan_angle() -> str:
    """Get current camera pan servo angle."""
    global _servo_angles
    return f"Camera pan angle: {_servo_angles['cam_pan']}°"

@function_tool
def get_cam_tilt_angle() -> str:
    """Get current camera tilt servo angle."""
    global _servo_angles
    return f"Camera tilt angle: {_servo_angles['cam_tilt']}°"

@function_tool
def get_ultrasonic_distance() -> str:
    """Get distance from ultrasonic sensor in centimeters."""
    try:
        px = get_picarx()
        distance = px.ultrasonic.read()
        return f"Ultrasound detected an obstacle at {distance:.1f} cm"
    except Exception as e:
        return f"Ultrasonic sensor error: {str(e)}"

@function_tool
def get_grayscale_data() -> str:
    """Get grayscale sensor readings (0-4095, left to right)."""
    try:
        px = get_picarx()
        data = px.get_grayscale_data()
        return f"Grayscale readings: Left={data[0]}, Center={data[1]}, Right={data[2]}"
    except Exception as e:
        return f"Grayscale sensor error: {str(e)}"

@function_tool
def get_line_status(val_list: List[int]) -> str:
    """Get line status from grayscale values."""
    try:
        px = get_picarx()
        status = px.get_line_status(val_list)
        line_detected = [i for i, detected in enumerate(status) if not detected]
        if line_detected:
            return f"Line detected on sensors: {line_detected}"
        else:
            return "No line detected"
    except Exception as e:
        return f"Line detection error: {str(e)}"

@function_tool
def get_cliff_status(val_list: List[int]) -> str:
    """Get cliff status from grayscale values."""
    try:
        px = get_picarx()
        is_cliff = px.get_cliff_status(val_list)
        return "Cliff detected - danger!" if is_cliff else "Ground detected - safe"
    except Exception as e:
        return f"Cliff detection error: {str(e)}"

@function_tool
def set_line_reference(refs: List[int]) -> str:
    """Set grayscale line reference values for line following calibration."""
    try:
        px = get_picarx()
        px.set_line_reference(refs)
        return f"Line reference values set: {refs}"
    except Exception as e:
        return f"Set line reference failed: {str(e)}"

@function_tool
def set_cliff_reference(refs: List[int]) -> str:
    """Set grayscale cliff reference values for cliff detection calibration."""
    try:
        px = get_picarx()
        px.set_cliff_reference(refs)
        return f"Cliff reference values set: {refs}"
    except Exception as e:
        return f"Set cliff reference failed: {str(e)}"

@function_tool
def init_camera() -> str:
    """Initialize the camera system with timeout protection."""
    global _vilib_initialized
    
    if _vilib_initialized:
        return "Camera already initialized"
    
    try:
        from vilib import Vilib
        Vilib.camera_start(vflip=False, hflip=False)
        Vilib.display(local=False, web=True)
        
        # Wait for camera with timeout (5 seconds)
        timeout = 5.0
        start_time = time.time()
        while time.time() - start_time < timeout:
            if hasattr(Vilib, 'flask_start') and Vilib.flask_start:
                time.sleep(0.5)  # Additional stabilization
                _vilib_initialized = True
                return "Camera initialized successfully"
            time.sleep(0.01)
        
        return "Camera initialization timeout - check camera connection"
        
    except Exception as e:
        return f"Camera initialization error: {str(e)}"

@function_tool
def capture_image(filename: str = "img_capture.jpg") -> str:
    """Capture an image from the camera and return confirmation with file path."""
    try:
        from vilib import Vilib
        import cv2
        
        if not _vilib_initialized:
            init_result = init_camera()
            if "error" in init_result.lower() or "timeout" in init_result.lower():
                return f"Camera capture failed: {init_result}"
        
        if hasattr(Vilib, 'img') and Vilib.img is not None:
            # Ensure filename has .jpg extension
            if not filename.endswith('.jpg'):
                filename += '.jpg'
            
            cv2.imwrite(filename, Vilib.img)
            abs_path = os.path.abspath(filename)
            return f"Image captured successfully: {abs_path}"
        else:
            return "Camera capture failed: No image available from camera"
            
    except Exception as e:
        return f"Camera capture error: {str(e)}"

@function_tool
def take_photo_vilib(name: str = None, path: str = "./") -> str:
    """Take a photo using Vilib's built-in photo function."""
    try:
        from vilib import Vilib
        
        if not _vilib_initialized:
            init_result = init_camera()
            if "error" in init_result.lower() or "timeout" in init_result.lower():
                return f"Photo capture failed: {init_result}"
        
        if name is None:
            from time import strftime, localtime
            name = f'photo_{strftime("%Y-%m-%d-%H-%M-%S", localtime(time.time()))}'
        
        Vilib.take_photo(name, path)
        full_path = os.path.abspath(f"{path}{name}.jpg")
        return f"Photo captured successfully: {full_path}"
        
    except Exception as e:
        return f"Photo capture error: {str(e)}"

@function_tool
def close_camera() -> str:
    """Close the camera system."""
    global _vilib_initialized
    if not _vilib_initialized:
        return "Camera not initialized"
    
    try:
        from vilib import Vilib
        Vilib.camera_close()
        _vilib_initialized = False
        return "Camera closed successfully"
    except Exception as e:
        return f"Camera close error: {str(e)}"

@function_tool
def play_sound(filename: str, volume: int = 100) -> str:
    """Play a sound file synchronously."""
    # Validate volume
    volume = int(clamp_value(volume, 0, 100))
    
    try:
        music = Music()
        music.sound_play(filename, volume)
        return f"Sound played: {filename} at {volume}% volume"
    except Exception as e:
        return f"Sound playback error: {str(e)}"

@function_tool
def play_sound_threading(filename: str, volume: int = 100) -> str:
    """Play a sound file asynchronously."""
    # Validate volume
    volume = int(clamp_value(volume, 0, 100))
    
    try:
        music = Music()
        music.sound_play_threading(filename, volume)
        return f"Sound started playing: {filename} at {volume}% volume"
    except Exception as e:
        return f"Sound playback error: {str(e)}"

@function_tool
def get_robot_status() -> str:
    """Get comprehensive robot status information."""
    try:
        px = get_picarx()
        status = {
            "servo_angles": _servo_angles,
            "camera_initialized": _vilib_initialized,
            "continuous_movement": _continuous_movement,
            "ultrasonic_distance": px.ultrasonic.read(),
            "grayscale_data": px.get_grayscale_data()
        }
        return json.dumps(status, indent=2)
    except Exception as e:
        return f"Status check error: {str(e)}"

@function_tool
def get_line_status_current() -> str:
    px = get_picarx()
    data = px.get_grayscale_data()
    status = px.get_line_status(data)
    return f"Grayscale={data} LineStatus={status}"


# Export all function tools for agent use
__all__ = [
    'reset', 'move_forward', 'move_backward', 'turn_left', 'turn_right', 
    'turn_in_place_left', 'turn_in_place_right', 'stop',
    'set_motor_speed', 'set_dir_servo_angle', 'set_cam_pan_angle', 'set_cam_tilt_angle',
    'get_servo_angles', 'get_dir_servo_angle', 'get_cam_pan_angle', 'get_cam_tilt_angle',
    'get_ultrasonic_distance', 'get_grayscale_data', 'get_line_status', 'get_cliff_status',
    'set_line_reference', 'set_cliff_reference', 'init_camera', 'capture_image',
    'take_photo_vilib', 'close_camera', 'play_sound', 'play_sound_threading', 'get_robot_status', 'get_line_status_current'
]