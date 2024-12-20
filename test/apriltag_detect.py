import cv2 
import apriltag
  

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

def show_camera(show_camera):
    window_title = "Camera"
    options = apriltag.DetectorOptions(families="tag36h11",nthreads=1, refine_edges=1, debug = 1)
    detector = apriltag.Detector(options)

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=0))
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        print('Camera started!')
        try:
            while True:
                ret_val, frame = video_capture.read()
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    results = detector.detect(gray)
                    print(results)
                except:
                    continue
                if show_camera:
                    window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
                    if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                        cv2.imshow(window_title, frame)
                    else:
                        break 
                keyCode = cv2.waitKey(10) & 0xFF
                # Stop the program on the ESC key or 'q'
                if keyCode == 27 or keyCode == ord('q') or keyCode == ord('Q') :
                    break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")



show_camera(True)