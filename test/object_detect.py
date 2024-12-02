import cv2
import numpy as np
import struct
import pickle
import socket
import threading

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=800,
    capture_height=600,
    display_width=800,
    display_height=600,
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

# cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

# if cap.isOpened():
#     print('Camera started!')
#     while True:
#         # Capture frame-by-frame
#         ret, frame = cap.read()

#         # Convert the frame to HSV color space
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#         # Define color range for the RC car (adjust these values as needed)
#         lower_color = np.array([35, 44, 83])
#         upper_color = np.array([84, 255, 255])

#         # Create a mask for the defined color
#         mask = cv2.inRange(hsv, lower_color, upper_color)

#         # Perform morphological operations to remove small noise
#         kernel = np.ones((5, 5), np.uint8)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

#         # Find contours in the mask
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         # Draw contours on the original frame
#         for contour in contours:
#             # Calculate the area of each contour
#             area = cv2.contourArea(contour)
            
#             # Filter out small contours (adjust the threshold as needed)
#             if area > 500:
#                 # Draw the contour on the frame
#                 cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
                
#                 # Draw a bounding box around the contour
#                 x, y, w, h = cv2.boundingRect(contour)
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

#         # Display the resulting frame
#         cv2.imshow('Frame', frame)

#         # Break the loop on 'q' key press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# # Release the video capture and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()

def video_capture():
    global frame
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec (e.g., XVID, MJPG, etc.)
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, (800, 600))

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if ret:
            # Write the frame to the video file
            out.write(frame)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release the VideoCapture and VideoWriter objects
    video_capture.release()
    out.release()

if __name__ == '__main__':
    video_thread = threading.Thread(target=video_capture)
    video_thread.start()
