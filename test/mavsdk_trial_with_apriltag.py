#!/usr/bin/env python3

import asyncio
from mavsdk import System
from mavsdk.offboard import (OffboardError, PositionNedYaw)
from pupil_apriltags import Detector
import cv2
import threading
import socket
import pickle
import struct
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
    capture_width=1920,
    capture_height=1080,
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

def object_detection(stream):#NEW
    global car_detected

    if stream:
        host = ''
        try:
            port = 8888

            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.bind((host, port))
            server_socket.listen(1)
            
            conn, addr = server_socket.accept()
        except:
            port = 8887

            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.bind((host, port))
            server_socket.listen(1)
            
            conn, addr = server_socket.accept()

    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec (e.g., XVID, MJPG, etc.)
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, (800, 600))

    if cap.isOpened():
        print('Camera started!')
        while True:
            ret, frame = cap.read()

            # Convert the frame to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Define color range for the RC car (adjust these values as needed)
            lower_color = np.array([19, 151, 116])
            upper_color = np.array([179, 255, 255])

            # Create a mask for the defined color
            mask = cv2.inRange(hsv, lower_color, upper_color)

            # Perform morphological operations to remove small noise
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw contours on the original frame
            for contour in contours:
                # Calculate the area of each contour
                area = cv2.contourArea(contour)
                
                # Filter out small contours (adjust the threshold as needed

                if area > 600:
                    car_detected = True
                    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
                    
                    # Draw a bounding box around the contour
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if stream:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                a, img_encoded = cv2.imencode('.jpg', frame, encode_param)
                data = np.array(img_encoded)
                string_data = data.tobytes()
                conn.sendall(struct.pack(">L", len(string_data)) + string_data)


            out.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    conn.close()
    server_socket.close()

def object_detection(stream):
    global car_detected

    if stream:
        host = ''
        try:
            port = 8888

            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.bind((host, port))
            server_socket.listen(1)
            
            conn, addr = server_socket.accept()
        except:
            port = 8887

            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.bind((host, port))
            server_socket.listen(1)
            
            conn, addr = server_socket.accept()

    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

    if cap.isOpened():
        print('Camera started!')
        while True:
            ret, frame = cap.read()

            #convert the frame to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            #define color range for the RC car (adjust these values as needed)
            lower_color = np.array([35, 44, 83])
            upper_color = np.array([84, 255, 255])

            #create a mask for the defined color
            mask = cv2.inRange(hsv, lower_color, upper_color)

            #perform morphological operations to remove small noise
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            #find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            #draw contours on the original frame
            for contour in contours:
                #calculate the area of each contour
                area = cv2.contourArea(contour)
                
                if area > 500: #area threshold
                    car_detected = True
                    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
                    
                    #draw a bounding box around the contour
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if stream:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                a, img_encoded = cv2.imencode('.jpg', frame, encode_param)
                data = np.array(img_encoded)
                string_data = data.tobytes()
                conn.sendall(struct.pack(">L", len(string_data)) + string_data)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    conn.close()
    server_socket.close()
        

def detect_apriltag(show_camera,stream):
    global apriltag_detected

    if stream:
        host = ''
        port = 8887

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((host, port))
        server_socket.listen(1)
        
        conn, addr = server_socket.accept()

    window_title = "Camera"
    at_detector = Detector(
    families="tag36h11"
    )
    c = 0
    #to flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        print('Camera started!')
        try:
            while True:
                ret_val, frame = video_capture.read()
                try:
                     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                     results = at_detector.detect(gray, True, camera_params, tag_size = 0.045)
                     if results != []:
                         apriltag_detected = True
                except:
                     continue
                
                #to display the camera in a window on the host machine
                if show_camera:
                    window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
                    if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                        cv2.imshow(window_title, frame)
                    else:
                        break
                
                #to stream the video on a TCP server:
                if stream:
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                    a, img_encoded = cv2.imencode('.jpg', frame, encode_param)
                    data = np.array(img_encoded)
                    string_data = data.tobytes()
                    conn.sendall(struct.pack(">L", len(string_data)) + string_data)


                keyCode = cv2.waitKey(10) & 0xFF
                if keyCode == 27 or keyCode == ord('q') or keyCode == ord('Q') :
                    break
                elif keyCode == ord('c'):
                    cv2.imwrite(f"frame{c}.jpg", frame)
                    c += 1
                    print('Image saved!')
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
            conn.close()
            server_socket.close()
    else:
        print("Error: Unable to open camera")

async def run():
    global apriltag_detected
    global car_detected

    drone = System()
    await drone.connect(system_address="udp://:4550")

    status_text_task = asyncio.ensure_future(print_status_text(drone))

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            break
    '''
    print("Waiting for drone to have a global position estimate...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("-- Global position estimate OK")
            break
    '''

    stream = False
    display_video = False

    apriltag_detected = False
    car_detected = False

    #detect_thread = threading.Thread(target=detect_apriltag, args=(display_video, stream))
    detect_thread = threading.Thread(target=object_detection, args=(stream,))
    detect_thread.start()

    print("-- Arming")
    await drone.action.arm()

    print("-- Taking off")
    await drone.action.takeoff()
    await asyncio.sleep(5)


    waypoints = [(-4.2, 3.6, -1, 90.0),(-4.2, 3.0, -1, 90.0),(-4.2, 2.4, -1, 90.0),(-4.2, 1.8, -1, 90.0),(-4.2, 2.4, -1, 90.0),(-4.2, 3.0, -1, 90.0),(-3.6, 3.6, -1, 90.0),(-3.0, 4.2, -1, 90.0)]

    for waypoint in waypoints:
        if not car_detected:
            await drone.offboard.set_position_ned(PositionNedYaw(waypoint[0], waypoint[1], waypoint[2], waypoint[3]))
            await asyncio.sleep(4)
        else:
            #print('Apriltag Detected!')
            print("car detected!")
            await drone.offboard.set_position_ned(PositionNedYaw(1.8, 4.2, -1, 90.0))
            await asyncio.sleep(2)
            break
    

    print("-- Landing")
    await drone.action.land()
    print('car detected:',car_detected)
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