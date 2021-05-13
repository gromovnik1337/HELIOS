#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(1280, 720)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Create a video encoder
ve = pipeline.createVideoEncoder()
ve.setDefaultProfilePreset(1920, 1080, 30, dai.VideoEncoderProperties.Profile.H265_MAIN)
cam_rgb.video.link(ve.input)

# Create output
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

veOut = pipeline.createXLinkOut()
veOut.setStreamName('veOut')
ve.bitstream.link(veOut.input)

fileColorH265 = open('color.h265', 'wb')
# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()
    print("Press q to stop encoding...")
    # Output queue will be used to get the rgb frames from the output defined above
    q_ve = device.getOutputQueue(name='veOut', maxSize=30, blocking=True)
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        in_rgb = q_rgb.get()  # blocking call, will wait until a new data has arrived
        # Retrieve 'bgr' (opencv format) frame
        cv2.imshow("bgr", in_rgb.getCvFrame())
        q_ve.get().getData().tofile(fileColorH265)



        if cv2.waitKey(1) == ord('q'):
            break

print("To view the encoded data, convert the stream file (.h264/.h265) into a video file (.mp4), using commands below:")
cmd = "ffmpeg -framerate 30 -i {} -c copy {}"
print(cmd.format("color.h265", "color.mp4"))
