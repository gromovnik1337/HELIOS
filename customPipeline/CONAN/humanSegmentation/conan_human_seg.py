#!/usr/bin/env python3

"""
Human segmentation standalone script using deeplabv3 model and rgb camera input
Created by: Team Sentinels, 03.04.2021
"""


from conan_human_seg_utils import *
import depthai as dai
import time
import numpy as np
import cv2

pipeline = dai.Pipeline()

# Definition of a pipeline
# ----------------------------------------------------------------------------
pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_2)

# This script works only with rgb input of the OAK - D camera
rgb = pipeline.createColorCamera()
rgb.setPreviewSize(nn_shape, nn_shape)
rgb.setInterleaved(False)

# Define a ANN node with deeplabv3_person_256 input
nn = pipeline.createNeuralNetwork()
nn.setBlobPath(nn_path)

nn.setNumPoolFrames(4)
nn.input.setBlocking(False) # Queue behavior when full
nn.setNumInferenceThreads(2) # Threads used by the node to run the inference 

# Link the preview output of the camera with the input to ANN
rgb.preview.link(nn.input)

rgb.setFps(40)

# Define outputs
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
# cam.preview.link(xout_rgb.input) # Send the raw frame to host
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")

# Overwrite default input queue behavior from blocking to overwriting
# ???
xout_rgb.input.setBlocking(False)
xout_nn.input.setBlocking(False)

# This links the passthrough message with the XLink output from the camera
# Why? Maybe because of the nature of the output from deeplab? Heatmap?
# N.B. Passthrough message is suitable for when the queue is set to overwriting
nn.passthrough.link(xout_rgb.input)

# Send the inference data to the host
nn.out.link(xout_nn.input)

# ----------------------------------------------------------------------------

device = dai.Device(pipeline)
device.startPipeline()

# Host side queues, parameters must correspond those from the pipeline inputs 
q_rgb = device.getOutputQueue(name = "rgb", maxSize = 4, blocking = False)
q_nn = device.getOutputQueue(name = "nn", maxSize = 4, blocking = False)

# Main program loop
# ----------------------------------------------------------------------------

# Auxiliary variables relevant for output
start_time = time.time()
counter = 0
fps = 0
layer_info_printed = False

while True:
    # Fetch latest results
    # tryGet() will return available data or return None otherwise
    in_rgb = q_rgb.get()
    in_nn = q_nn.get()

    # RGB camera input (1D array) conversion into Height-Width-Channels (HWC) form
    if in_rgb is not None:
        shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
        frame = in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
        frame = np.ascontiguousarray(frame)

    # ANN results (1D array, fixed size, no matter how much results ANN has produced, results end with -1, the rest is filled with 0s) transformations 
    if in_nn is not None:
        # print("NN received")
        # https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.NNData.getAllLayers
        # https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.TensorInfo
        layers = in_nn.getAllLayers()

        # Print the info acquired from the deepplabv3
        if not layer_info_printed:
            for layer_nr, layer in enumerate(layers):
                print(f"Layer {layer_nr}")
                print(f"Name: {layer.name}")
                print(f"Order: {layer.order}")
                print(f"dataType: {layer.dataType}")
                dims = layer.dims[::-1] # reverse dimensions
                print(f"dims: {dims}")
            layer_info_printed = True

        # Relevant information is in layer1
        # getLayerInt32 takes layer name as an argument it is different from getFirstLayerInt32
        layer_1 = in_nn.getLayerInt32(layers[0].name)

        # Create numpy output
        dims = layer.dims[::-1]
        layer_1 = np.asarray(layer_1, dtype=np.int32).reshape(dims)

        output_colors = decode_deeplabv3p(layer_1)    

        if frame is not None:
            frame = show_deeplabv3p(output_colors, frame)
            cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
            cv2.imshow("nn_input", frame)
    
    counter+=1
    if (time.time() - start_time) > 1 :
        fps = counter / (time.time() - start_time)

        counter = 0
        start_time = time.time()


    if cv2.waitKey(1) == ord('q'):
        break



    

