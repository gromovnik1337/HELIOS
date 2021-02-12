import cv2
import numpy as np

# Open or read the images
inputImage = cv2.imread('imageCode.jpg')

# Resize the images to speed up processing
inputImage = cv2.resize(inputImage,(640,480))

# Convert images to grayscale, to simplify the dataset.
inputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

# Create a QR decoder object
qrDecoder = cv2.QRCodeDetector()

# Display barcode and QR code location
def display(im, bbox):
    n = len(bbox)
    for j in range(n):
        cv2.line(im, tuple(bbox[j][0]), tuple(bbox[ (j+1) % n][0]), (255,0,0), 3)

    # Display results
    cv2.imshow("Results", im)

# Detect and decode the QRcode
data,bbox,rectifiedImage = qrDecoder.detectAndDecode(inputImage)
if len(data)>0:
    print("Decoded Data : {}".format(data))
    display(inputImage, bbox)
    rectifiedImage = np.uint8(rectifiedImage);
    cv2.imshow("Rectified QRCode", rectifiedImage);
else:
    print("QR Code not detected")
    cv2.imshow("Results", inputImage)

# Wait for key press and then terminate all open windows
cv2.waitKey(0)
cv2.destroyAllWindows()