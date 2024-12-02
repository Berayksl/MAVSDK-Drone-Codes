#SCRIPT FOR FINDING THE HSV VALUES OF THE OBJECT TO BE DETECTED

import cv2
import numpy as np

def nothing(x):
    pass

# Load a sample image of your RC car
image_path = '/home/nano/Desktop/Drone Codes/frame.jpg'
image = cv2.imread(image_path)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Create a window
cv2.namedWindow('image')

# Create trackbars for color change
cv2.createTrackbar('H Lower', 'image', 0, 179, nothing)
cv2.createTrackbar('S Lower', 'image', 0, 255, nothing)
cv2.createTrackbar('V Lower', 'image', 0, 255, nothing)
cv2.createTrackbar('H Upper', 'image', 0, 179, nothing)
cv2.createTrackbar('S Upper', 'image', 0, 255, nothing)
cv2.createTrackbar('V Upper', 'image', 0, 255, nothing)

# Initialize the upper and lower HSV range
cv2.setTrackbarPos('H Upper', 'image', 179)
cv2.setTrackbarPos('S Upper', 'image', 255)
cv2.setTrackbarPos('V Upper', 'image', 255)

while True:
    # Get the current positions of the trackbars
    h_lower = cv2.getTrackbarPos('H Lower', 'image')
    s_lower = cv2.getTrackbarPos('S Lower', 'image')
    v_lower = cv2.getTrackbarPos('V Lower', 'image')
    h_upper = cv2.getTrackbarPos('H Upper', 'image')
    s_upper = cv2.getTrackbarPos('S Upper', 'image')
    v_upper = cv2.getTrackbarPos('V Upper', 'image')

    # Define the HSV range for masking
    lower_color = np.array([h_lower, s_lower, v_lower])
    upper_color = np.array([h_upper, s_upper, v_upper])

    # Create the mask
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Apply the mask to the original image
    result = cv2.bitwise_and(image, image, mask=mask)

    # Show the original image, mask, and the result
    cv2.imshow('Original', image)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
