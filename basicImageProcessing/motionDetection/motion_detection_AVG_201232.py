"""
This script detects the object inside the continous video frame.
Created by: Vice, 30.12.2020 (from the source: https://github.com/bnbe-club/opencv-object-detection-diy-33)
"""
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np

# Initialize the camera, set the basic parameters, grab a reference to the raw camera output
camera = PiCamera()

camera.resolution = (640, 480)
camera.framerate = 30
camera.rotation = 180

rawCapture = PiRGBArray(camera, size = (640, 480))
avg = None

time.sleep(2) # To allow the camera to adjust to lighting/white balance

# Initiate video or frame capture sequence
for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    # Grab the raw array representation of the image & save it for later use
    frame = f.array
    frameRef = frame
    
    # Convert the current frame to grayscale &  blur the result
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (21, 21), 0)
    
    # Initialize the avg if it hasn't been done
    if avg is None:
        avg = frame.copy().astype("float")
        rawCapture.truncate(0) # This clears the stream
        continue
    
    # Accumulate the weighted average between the current frame and
    # previous frames, then compute the difference between the current
    # frame and running average
    cv2.accumulateWeighted(frame, avg, 0.05)
    frameDelta = cv2.absdiff(frame, cv2.convertScaleAbs(avg))

    # Convert the difference into binary image
    # Dilate the thresholded image to fill in holes (grouping nearby pixels)
    frameDelta = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    frameDelta = cv2.dilate(frameDelta, None, iterations=2)

    # Find contours or continuous white clusters(blobs) in the image
    contours, hierarchy = cv2.findContours(frameDelta, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the index of the largest contour = moving object
    if len(contours) > 0:
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        contoursMax = contours[max_index]   

        # Draw a bounding box/rectangle around the object
        x,y,w,h = cv2.boundingRect(contoursMax)
        cv2.rectangle(frameRef, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frameRef, "Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    # Show the frame
    cv2.imshow("Video", frameRef)   

    # Clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # If the 'q' key is pressed then break from the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
cv2.destroyAllWindows()