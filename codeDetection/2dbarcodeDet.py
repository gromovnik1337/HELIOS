# import the necessary packages
from pylibdmtx.pylibdmtx import decode

import cv2
import numpy as np

# Open or read the images
inputImage = cv2.imread('imageCode.jpg')

# Resize the images to speed up processing
inputImage = cv2.resize(inputImage,(640,480))

# Convert images to grayscale, to simplify the dataset.
inputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

# Thershold the image
ret,thresh = cv2.threshold(inputImage, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

result = decode(thresh)

print(result[0].data)

x = result[0].rect.left
y = result[0].rect.top
w = result[0].rect.width
h = result[0].rect.height
cv2.rectangle(inputImage, (x,y), (x+w,y+h), (0,255,0), 2)
cv2.imshow("title",inputImage)

# Wait for key press and then terminate all open windows
cv2.waitKey(0)
cv2.destroyAllWindows()
