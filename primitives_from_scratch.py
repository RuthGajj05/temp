# First attempt to implement primitives
from picarx import Picarx
import time

px = Picarx()

def move_forward(speed, duration):
    px.forward(speed)
    time.sleep(duration)
    px.stop()

def move_backward(speed, duration):
    px.backward(speed)
    time.sleep(duration)
    px.stop()

def turn_left(speed, duration, angle):
    px.set_dir_servo_angle(-angle)
    px.forward(speed)
    time.sleep(duration)
    px.stop()
    px.set_dir_servo_angle(0)

def turn_right(speed, duration, angle):
    px.set_dir_servo_angle(angle)
    px.forward(speed)
    time.sleep(duration)
    px.stop()
    px.set_dir_servo_angle(0)

def turn_in_place_left(speed, duration, angle):
    px.set_dir_servo_angle(0)
    px.set_motor_speed(1, -speed) # 1 = left, 2 = right
    px.set_motor_speed(2,  speed)
    time.sleep(float(duration))
    px.stop()

def turn_in_place_right(speed, duration, angle):
    px.set_dir_servo_angle(0)
    px.set_motor_speed(1,  speed)
    px.set_motor_speed(2, -speed)
    time.sleep(float(duration))
    px.stop()

def stop():
    px.stop()

def set_direction_servo_angle(angle):
    px.set_dir_servo_angle(angle)

def set_camera_pan_angle(angle):
    px.set_cam_pan_angle(angle)


def set_camera_tilt_angle(angle):
    px.set_cam_tilt_angle(angle)

def get_ultrasonic_distance():
    return px.ultrasonic.read()

def get_grayscale_data():
    return px.get_grayscale_data()

def get_line_status(val_list):
    return px.get_line_status(val_list)

def get_cliff_status(val_list):
    return px.get_cliff_status(val_list)

def set_line_reference(val_list):
    px.set_line_reference(val_list)

def init_camera(vflip=False, hflip=False):
    from vilib import Vilib
    Vilib.camera_start(vflip=vflip, hflip=hflip)
    Vilib.display(local=False, web=True)
    t0 = time.time()
    while time.time() - t0 < 5:
        if getattr(Vilib, "flask_start", False):
            return True
        time.sleep(0.01)
    return False

def capture_image(filename):
    from vilib import Vilib
    import cv2
    if not getattr(Vilib, "img", None):
        if not init_camera():
            return False
    return cv2.imwrite(filename, Vilib.img)

def close_camera():
    from vilib import Vilib
    Vilib.camera_close()

def reset():
    px.stop()
    px.set_dir_servo_angle(0)
    try:
        px.set_cam_pan_angle(0)
        px.set_cam_tilt_angle(0)
    except Exception:
        pass
    px.set_motor_speed(1, 0)
    px.set_motor_speed(2, 0)
