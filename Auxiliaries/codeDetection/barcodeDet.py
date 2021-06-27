# import the necessary packages
import pyzbar
from pyzbar.pyzbar import decode
import argparse

import cv2
import numpy as np
"""
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())
"""
# Open or read the images
inputImage = cv2.imread('imageCode.jpg')

# Resize the images to speed up processing
inputImage = cv2.resize(inputImage,(640,480))

# Convert images to grayscale, to simplify the dataset.
inputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

barcodes = decode(inputImage)

# loop over the detected barcodes
for barcode in barcodes:
    # extract the bounding box location of the barcode and draw the
    # bounding box surrounding the barcode on the image
    (x, y, w, h) = barcode.rect
    cv2.rectangle(inputImage, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # the barcode data is a bytes object so if we want to draw it on
    # our output image we need to convert it to a string first
    barcodeData = barcode.data.decode("utf-8")
    barcodeType = barcode.type
    # draw the barcode data and barcode type on the image
    text = "{} ({})".format(barcodeData, barcodeType)
    cv2.putText(inputImage, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (0, 0, 255), 2)
    # print the barcode type and data to the terminal
    print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))




# Wait for key press and then terminate all open windows
cv2.waitKey(0)
cv2.destroyAllWindows()
