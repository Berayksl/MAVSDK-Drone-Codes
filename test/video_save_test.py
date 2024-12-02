import cv2
import threading

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=800,
    capture_height=600,
    display_width=800,
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

def video_capture():
    global frame
    # Use the appropriate gstreamer pipeline or a default camera source
    video_capture = cv2.VideoCapture(0)  # Change to gstreamer_pipeline(flip_method=0) if required
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec (e.g., XVID, MJPG, etc.)
    
    # Get the width and height of the frame from the camera source
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter('output.avi', fourcc, 30.0, (width, height))

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if ret:
            # Write the frame to the video file
            out.write(frame)
            #cv2.imshow('Frame', frame)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Failed to capture frame")
            break

    # Release the VideoCapture and VideoWriter objects
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_thread = threading.Thread(target=video_capture)
    video_thread.start()
