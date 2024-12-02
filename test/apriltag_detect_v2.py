import cv2 
#import apriltag
from pupil_apriltags import Detector
import pickle
import socket
import numpy as np
import struct



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

def detect_apriltag(show_camera,stream):


    if stream:
        host = ''
        port = 8888

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((host, port))
        server_socket.listen(1)
        
        conn, addr = server_socket.accept()

    window_title = "Camera"
    at_detector = Detector(
    families="tag36h11"
    )
    c = 0
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        print('Camera started!')
        try:
            while True:
                ret_val, frame = video_capture.read()
                #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                try:
                     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                     results = at_detector.detect(gray, True, camera_params, tag_size = 0.045)
                     #print(results)
                except:
                     continue
                if show_camera:
                    window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
                    if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                        cv2.imshow(window_title, frame)
                    else:
                        break

                if stream:
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                    result, img_encoded = cv2.imencode('.jpg', frame, encode_param)
                    data = np.array(img_encoded)
                    string_data = data.tobytes()
                    conn.sendall(struct.pack(">L", len(string_data)) + string_data)
                    if (c % 120) ==0:
                        cv2.imwrite(f"frame{c}.jpg", frame)
                        
                        print('Image saved!')
                    c += 1



                keyCode = cv2.waitKey(10) & 0xFF
                # Stop the program on the ESC key or 'q'

                     
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



detect_apriltag(False, True)