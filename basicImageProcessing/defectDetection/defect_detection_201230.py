"""
This script compares two images and finds the difference which is assumed to be a defect.
Created by: Vice, 30.12.2020 (from the source: https://github.com/bnbe-club/opencv-object-detection-diy-33)
"""

import cv2
import numpy as np

# Read the referrent image
imgRef = cv2.imread('imageRef.jpg')

# Read the image with the defect
imgDef = cv2.imread('imageDef.jpg')

# Resize the images to speed up processing
imgRef = cv2.resize(imgRef,(640,480))
imgDef = cv2.resize(imgDef,(640,480))

# Save the images for later use 
imgRefRef = imgRef
imgDefRef = imgDef

# Convert images to grayscale, to simplify the dataset.
imgRef = cv2.cvtColor(imgRef, cv2.COLOR_BGR2GRAY)
imgDef = cv2.cvtColor(imgDef, cv2.COLOR_BGR2GRAY)

# Blur the images to get rid of sharp edges/outlines (further simplification)
imgRef = cv2.GaussianBlur(imgRef, (21, 21), 0)
imgDef = cv2.GaussianBlur(imgDef, (21, 21), 0)

# Obtain the difference between the two images
imgDelta = cv2.absdiff(imgRef, imgDef)

# Convert the difference into binary image
imgDelta = cv2.threshold(imgDelta, 25, 255, cv2.THRESH_BINARY)[1]

# Dilate the thresholded image to fill in holes (grouping nearby pixels)
imgDelta = cv2.dilate(imgDelta, None, iterations=2)

# Find contours or continuous white clusters(blobs) in the image
contours, hierarchy = cv2.findContours(imgDelta, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Find the index of the largest contour = defect
areas = [cv2.contourArea(c) for c in contours]
max_index = np.argmax(areas)
contoursMax = contours[max_index]

# Draw a bounding box/rectangle around the defect
x,y,w,h = cv2.boundingRect(contoursMax)
cv2.rectangle(imgDefRef, (x,y), (x+w,y+h), (0,255,0), 2)
cv2.putText(imgDefRef, "Defect", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the images for comparison
cv2.imshow("Referent image", imgRefRef)
cv2.imshow("Image with the defect", imgDefRef)

# Wait for a key being pressed & terminate all open windows
cv2.waitKey(0)
cv2.destroyAllWindows()
