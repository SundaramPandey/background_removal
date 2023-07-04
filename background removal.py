import cv2
import numpy as np

# Attach camera indexed as 0
camera = cv2.VideoCapture(0)

# Set frame width and height as 640 x 480
camera.set(3, 640)
camera.set(4, 480)

# Load the mountain image
mountain = cv2.imread('mount_everest.jpg')
# Resize the mountain image as 640 x 480
mountain = cv2.resize(mountain, (640, 480))

while True:
    # Read a frame from the attached camera
    status, frame = camera.read()

    # If we got the frame successfully
    if status:
        # Flip it
        frame = cv2.flip(frame, 1)

        # Converting the image to RGB for easy processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Creating thresholds for blue color range
        lower_bound = np.array([100, 50, 50])
        upper_bound = np.array([140, 255, 255])
        mask = cv2.inRange(frame_rgb, lower_bound, upper_bound)

        # Apply morphological operations to the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

        # Invert the mask
        mask_2 = cv2.bitwise_not(mask)

        # Use the mask to extract the foreground (person) and the background (mountain)
        foreground = cv2.bitwise_and(frame, frame, mask=mask_2)
        background = cv2.bitwise_and(mountain, mountain, mask=mask)

        # Combine the foreground and background to get the final output
        final_output = cv2.addWeighted(foreground,1, background,1,0)

        # Display the final output
        cv2.imshow("Magic", final_output)

        # Display the original frame
        cv2.imshow('Frame', frame)

        # Wait for 1ms before displaying another frame
        code = cv2.waitKey(1)
        if code == 32:
            break

# release the camera and close all opened windows
camera.release()
cv2.destroyAllWindows()
