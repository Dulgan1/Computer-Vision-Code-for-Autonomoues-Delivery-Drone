import math
import time
from pymavlink import mavutil

# --- 1080p CAMERA PARAMETERS ---
IMG_WIDTH = 1920.0
IMG_HEIGHT = 1080.0

# Replace with your specific wide-angle lens FOV specifications
HFOV_DEG = 67.5 
VFOV_DEG = 42.5

# Pre-calculate center points and FOV constants for loop efficiency
CX = IMG_WIDTH / 2.0
CY = IMG_HEIGHT / 2.0
TAN_HALF_HFOV = math.tan(math.radians(HFOV_DEG) / 2.0)
TAN_HALF_VFOV = math.tan(math.radians(VFOV_DEG) / 2.0)

'''try:
    pixhawk = mavutil.mavlink_connection('/dev/ttyAMA0', baud=921600)
    pixhawk.wait_heartbeat(timeout=5)
    print("MAVLink connection established.")
except Exception as e:
    print(f"Failed to connect to Pixhawk: {e}")
'''

def send_landing_target_pixels(pixhawk, marker_pixel_list):
    """
    Converts absolute [x, y] pixel coordinates to angular offsets.
    Streams to ArduPilot using the FRD coordinate frame.
    """
    marker_x = marker_pixel_list[0]
    marker_y = marker_pixel_list[1]

    # 1. Calculate pixel offsets from the optical center
    offset_x = marker_x - CX
    offset_y = marker_y - CY

    # 2. Axis Alignment to MAV_FRAME_BODY_FRD
    # Assuming the "top" of the camera frame points to the drone's nose
    drone_x_px = -offset_y  # Forward is negative Y in OpenCV
    drone_y_px = offset_x   # Right is positive X in OpenCV

    # 3. Normalize against the center and apply Trigonometric FOV Conversion
    angle_x = math.atan((drone_x_px / CY) * TAN_HALF_VFOV)
    angle_y = math.atan((drone_y_px / CX) * TAN_HALF_HFOV)

    # 4. Stream MAVLink Message
    pixhawk.mav.landing_target_send(
        time_usec=int(time.time() * 1e6),
        target_num=0,
        frame=mavutil.mavlink.MAV_FRAME_BODY_FRD, 
        angle_x=angle_x,
        angle_y=angle_y,
        distance=0.0,  # Forces ArduPilot to use its internal altitude estimate
        size_x=0.0,
        size_y=0.0
    )


def drop_payload(pixhawk, servo_channel, servo_open_pwm):
    """Actuates the servo mechanism."""
    pixhawk.mav.command_long_send(
        pixhawk.target_system, pixhawk.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO, 0,
        servo_channel, servo_open_pwm, 0, 0, 0, 0, 0
    )
    print("PAYLOAD DROPPED!")