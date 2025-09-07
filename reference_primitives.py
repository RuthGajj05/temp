# Import the main PicarX robot control class
from picarx import Picarx
# Import time module for sleep delays
import time
# Import type hints for better code documentation
from typing import List, Optional
# Import Music class for sound playback functionality
from robot_hat import Music
# Import os module for file operations
import os

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

def get_picarx() -> Picarx:
    # Access the global PicarX instance variable
    global _picarx
    # Check if PicarX instance doesn't exist yet
    if _picarx is None:
        # Create new PicarX instance and assign to global variable
        _picarx = Picarx()
    # Return the PicarX instance (either existing or newly created)
    return _picarx

def reset() -> None:
    # Access the global servo angles dictionary
    global _servo_angles
    # Get the PicarX instance
    px = get_picarx()
    # Set direction servo to center position (0 degrees)
    px.set_dir_servo_angle(0)
    # Set camera pan servo to center position (0 degrees)
    px.set_cam_pan_angle(0)
    # Set camera tilt servo to center position (0 degrees)
    px.set_cam_tilt_angle(0)
    # Stop all motor movement
    px.stop()
    # Update global servo angles to reflect center positions
    _servo_angles['dir_servo'] = 0
    _servo_angles['cam_pan'] = 0
    _servo_angles['cam_tilt'] = 0

def move_forward(speed: int, duration: float) -> None:
    # Access the global servo angles dictionary
    global _servo_angles
    # Get the PicarX instance
    px = get_picarx()
    # Start moving forward at specified speed
    px.forward(speed)
    # Wait for the specified duration
    time.sleep(duration)
    # Stop the robot
    px.stop()
    # Reset direction servo angle to 0 (straight)
    _servo_angles['dir_servo'] = 0

def move_backward(speed: int, duration: float) -> None:
    # Access the global servo angles dictionary
    global _servo_angles
    # Get the PicarX instance
    px = get_picarx()
    # Start moving backward at specified speed
    px.backward(speed)
    # Wait for the specified duration
    time.sleep(duration)
    # Stop the robot
    px.stop()
    # Reset direction servo angle to 0 (straight)
    _servo_angles['dir_servo'] = 0

def turn_left(angle: float, speed: int, duration: float) -> None:
    # Access the global servo angles dictionary
    global _servo_angles
    # Get the PicarX instance
    px = get_picarx()
    # Set motor calibration values for left turn: left motor backward, right motor forward
    px.cali_dir_value = [-1, 1]
    # Start moving forward (which will cause left turn due to motor calibration)
    px.forward(speed)
    # Wait for the specified duration
    time.sleep(duration)
    # Stop the robot
    px.stop()
    # Reset motor calibration values to normal forward movement
    px.cali_dir_value = [1, 1]
    # Reset direction servo angle to 0 (straight)
    _servo_angles['dir_servo'] = 0

def turn_right(angle: float, speed: int, duration: float) -> None:
    # Access the global servo angles dictionary
    global _servo_angles
    # Get the PicarX instance
    px = get_picarx()
    # Set motor calibration values for right turn: left motor forward, right motor backward
    px.cali_dir_value = [1, -1]
    # Start moving forward (which will cause right turn due to motor calibration)
    px.forward(speed)
    # Wait for the specified duration
    time.sleep(duration)
    # Stop the robot
    px.stop()
    # Reset motor calibration values to normal forward movement
    px.cali_dir_value = [1, 1]
    # Reset direction servo angle to 0 (straight)
    _servo_angles['dir_servo'] = 0

def stop() -> None:
    # Get the PicarX instance
    px = get_picarx()
    # Stop all motor movement immediately
    px.stop()

def set_motor_speed(motor_id: int, speed: int) -> None:
    # Get the PicarX instance
    px = get_picarx()
    # Set individual motor speed (1=left motor, 2=right motor, speed=-100 to 100)
    px.set_motor_speed(motor_id, speed)

def set_dir_servo_angle(angle: float) -> None:
    # Access the global servo angles dictionary
    global _servo_angles
    # Get the PicarX instance
    px = get_picarx()
    # Clamp angle to maximum 30 degrees
    if angle > 30:
        angle = 30
    # Clamp angle to minimum -30 degrees
    if angle < -30:
        angle = -30
    # Set the direction servo to the specified angle
    px.set_dir_servo_angle(angle)
    # Update global servo angles to track current position
    _servo_angles['dir_servo'] = angle

def set_cam_pan_angle(angle: float) -> None:
    # Access the global servo angles dictionary
    global _servo_angles
    # Get the PicarX instance
    px = get_picarx()
    # Set the camera pan servo to the specified angle
    px.set_cam_pan_angle(angle)
    # Update global servo angles to track current position
    _servo_angles['cam_pan'] = angle

def set_cam_tilt_angle(angle: float) -> None:
    # Access the global servo angles dictionary
    global _servo_angles
    # Get the PicarX instance
    px = get_picarx()
    # Set the camera tilt servo to the specified angle
    px.set_cam_tilt_angle(angle)
    # Update global servo angles to track current position
    _servo_angles['cam_tilt'] = angle

def get_servo_angles() -> dict:
    # Access the global servo angles dictionary
    global _servo_angles
    # Return a copy of the servo angles dictionary
    return _servo_angles.copy()

def get_dir_servo_angle() -> float:
    # Access the global servo angles dictionary
    global _servo_angles
    # Return the current direction servo angle
    return _servo_angles['dir_servo']

def get_cam_pan_angle() -> float:
    # Access the global servo angles dictionary
    global _servo_angles
    # Return the current camera pan servo angle
    return _servo_angles['cam_pan']

def get_cam_tilt_angle() -> float:
    # Access the global servo angles dictionary
    global _servo_angles
    # Return the current camera tilt servo angle
    return _servo_angles['cam_tilt']

def get_ultrasonic_distance() -> float:
    # Get the PicarX instance
    px = get_picarx()
    # Read distance from ultrasonic sensor and return in centimeters
    return px.ultrasonic.read()

def get_grayscale_data() -> List[int]:
    # Get the PicarX instance
    px = get_picarx()
    # Get grayscale sensor readings (0-4095, left to right) and return as list
    return px.get_grayscale_data()

def get_line_status(val_list: List[int]) -> List[bool]:
    # Get the PicarX instance
    px = get_picarx()
    # Determine line status from grayscale values and return as list of booleans
    return px.get_line_status(val_list)

def get_cliff_status(val_list: List[int]) -> bool:
    # Get the PicarX instance
    px = get_picarx()
    # Determine cliff status from grayscale values and return boolean
    return px.get_cliff_status(val_list)

def set_line_reference(refs: List[int]) -> None:
    # Get the PicarX instance
    px = get_picarx()
    # Set grayscale line reference values for line following calibration
    px.set_line_reference(refs)

def set_cliff_reference(refs: List[int]) -> None:
    # Get the PicarX instance
    px = get_picarx()
    # Set grayscale cliff reference values for cliff detection calibration
    px.set_cliff_reference(refs)

def init_camera() -> None:
    # Access the global camera initialization flag
    global _vilib_initialized
    # Check if camera hasn't been initialized yet
    if not _vilib_initialized:
        # Try to initialize camera system
        try:
            # Import Vilib camera library
            from vilib import Vilib
            # Start camera with no vertical or horizontal flip
            Vilib.camera_start(vflip=False, hflip=False)
            # Start camera display (local=False, web=True)
            Vilib.display(local=False, web=True)
            # Wait for camera to be ready by checking flask_start property
            while True:
                # Check if Vilib has flask_start attribute and it's True
                if hasattr(Vilib, 'flask_start') and Vilib.flask_start:
                    # Camera is ready, break out of loop
                    break
                # Wait 10ms before checking again
                time.sleep(0.01)
            # Additional stabilization time
            time.sleep(0.5)
            # Set camera initialization flag to True
            _vilib_initialized = True
            # Print success message
            print("Camera initialized successfully")
        # Handle any camera initialization errors
        except Exception as e:
            # Print error message
            print(f"Camera initialization error: {e}")

def capture_image(filename: str = "img_capture.jpg") -> None:
    # Try to capture an image
    try:
        # Import Vilib camera library
        from vilib import Vilib
        # Import OpenCV for image processing
        import cv2
        # Check if camera hasn't been initialized
        if not _vilib_initialized:
            # Initialize camera first
            init_camera()
        # Check if Vilib has img attribute and image is available
        if hasattr(Vilib, 'img') and Vilib.img is not None:
            # Save the current camera image to specified filename
            cv2.imwrite(filename, Vilib.img)
            # Print success message
            print(f"Image saved as {filename}")
        else:
            # Print message if no image is available
            print("No image available from camera")
    # Handle any camera capture errors
    except Exception as e:
        # Print error message
        print(f"Camera capture error: {e}")

def take_photo_vilib(name: str = None, path: str = "./") -> str:
    # Try to take a photo using Vilib's built-in function
    try:
        # Import Vilib camera library
        from vilib import Vilib
        # Import time module for timestamp generation
        import time
        # Check if camera hasn't been initialized
        if not _vilib_initialized:
            # Initialize camera first
            init_camera()
        # Check if no name was provided
        if name is None:
            # Import time formatting functions
            from time import strftime, localtime
            # Generate timestamp-based filename
            name = f'photo_{strftime("%Y-%m-%d-%H-%M-%S", localtime(time.time()))}'
        # Take photo using Vilib's built-in function
        Vilib.take_photo(name, path)
        # Create full path for the saved photo
        full_path = f"{path}{name}.jpg"
        # Print success message with full path
        print(f'Photo saved as {full_path}')
        # Return the full path of the saved photo
        return full_path
    # Handle any photo capture errors
    except Exception as e:
        # Print error message
        print(f"Photo capture error: {e}")
        # Return empty string to indicate failure
        return ""

def close_camera() -> None:
    # Access the global camera initialization flag
    global _vilib_initialized
    # Check if camera has been initialized
    if _vilib_initialized:
        # Try to close camera system
        try:
            # Import Vilib camera library
            from vilib import Vilib
            # Close the camera system
            Vilib.camera_close()
            # Set camera initialization flag to False
            _vilib_initialized = False
            # Print success message
            print("Camera closed")
        # Handle any camera close errors
        except Exception as e:
            # Print error message
            print(f"Camera close error: {e}")

def play_sound(filename: str, volume: int = 100) -> None:
    # Try to play a sound file
    try:
        # Create Music instance for sound playback
        music = Music()
        # Play sound file synchronously (blocks until finished)
        music.sound_play(filename, volume)
    # Handle any sound playback errors
    except Exception as e:
        # Print error message
        print(f"Sound playback error: {e}")

def play_sound_threading(filename: str, volume: int = 100) -> None:
    # Try to play a sound file asynchronously
    try:
        # Create Music instance for sound playback
        music = Music()
        # Play sound file asynchronously (non-blocking)
        music.sound_play_threading(filename, volume)
    # Handle any sound playback errors
    except Exception as e:
        # Print error message
        print(f"Sound playback error: {e}")