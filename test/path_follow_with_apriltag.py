#Drone follows the given waypoints and lands if it detects an apriltag

#!/usr/bin/env python3

import asyncio
from mavsdk import System
from mavsdk.offboard import (OffboardError, PositionNedYaw)
from pupil_apriltags import Detector
import pickle
import cv2

import numpy as np

with open('/home/nano/Desktop/Drone Codes/cameraMatrix.pkl', 'rb') as file:
          cameraMatrix = pickle.load(file)
camera_params = (
    cameraMatrix[0, 0],
    cameraMatrix[1, 1],
    cameraMatrix[0, 2],
    cameraMatrix[1, 2],
    
)          
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=800,
    capture_height=600,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


async def run():
    at_detector = Detector(
    families="tag36h11"
    )
    c = 0
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))

    drone = System()
    print('here')
    await drone.connect(system_address="udp://:4550")

    status_text_task = asyncio.ensure_future(print_status_text(drone))

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            break
    
    # print("Waiting for drone to have a global position estimate...")
    # async for health in drone.telemetry.health():
    #     if health.is_global_position_ok and health.is_home_position_ok:
    #         print("-- Global position estimate OK")
    #         break
    
    print("-- Arming")
    await drone.action.arm()
    
    print("-- Taking off")
    await drone.action.takeoff()
    await asyncio.sleep(5)

    await drone.offboard.set_position_ned(
    PositionNedYaw(0.0, 0.0, -1, 90.0))
    await asyncio.sleep(5)

    await drone.offboard.set_position_ned(
    PositionNedYaw(0.0, 1.75, -1, 90.0))
    await asyncio.sleep(5)

    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    ret_val, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = at_detector.detect(gray, True, camera_params, tag_size = 0.045)
    cv2.imwrite("test_image.jpg", frame)
    print(results)
    if results != []:
        print('detected')
        print(results)
        print("-- Landing")
        await drone.action.land()
        status_text_task.cancel()
    else:
        await drone.offboard.set_position_ned(
        PositionNedYaw(1.5, 1.50, -1, 90.0))
        await asyncio.sleep(5)
        await drone.offboard.set_position_ned(
        PositionNedYaw(1.5, 0, -1, 90.0))
        await asyncio.sleep(5)
        await drone.offboard.set_position_ned(
        PositionNedYaw(0, 0, -1, 90.0))
        await asyncio.sleep(5)
        print("-- Landing")
        await drone.action.land()

        status_text_task.cancel()


async def print_status_text(drone):
    try:
        async for status_text in drone.telemetry.status_text():
            print(f"Status: {status_text.type}: {status_text.text}")
    except asyncio.CancelledError:
        return


if __name__ == "__main__":
    # Run the asyncio loop
    asyncio.run(run())