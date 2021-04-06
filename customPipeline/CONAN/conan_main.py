#!/usr/bin/env python3

from conan_utils import *
import depthai as dai
import numpy as np
from imutils.video import FPS
import threading
import time
import cv2

# Definition of a pipeline
# ----------------------------------------------------------------------------
def create_pipeline():
    pipeline = dai.Pipeline()
    # Copy pasted from human segmentation file, reason unknown :)
    pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_2)

    # Define camera object compatible with deeplabv3_person_513 input
    rgb = pipeline.createColorCamera()
    rgb.setPreviewSize(nn_shape_1, nn_shape_1)
    rgb.setInterleaved(False)
    
    # Define a ANN node with deeplabv3_person_513 input
    nn_1 = pipeline.createNeuralNetwork()
    nn_1.setBlobPath(nn_path_1)

    # Parameters relevant for deepblabv3
    nn_1.setNumPoolFrames(4)
    nn_1.input.setBlocking(False) # Queue behavior when full
    nn_1.setNumInferenceThreads(2) # Threads used by the node to run the inference

    # Link the preview of the camera with the input to ANN
    rgb.preview.link(nn_1.input)

    rgb.setFps(40)

    # Define outputs
    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb_stream")

    xout_nn_1 = pipeline.createXLinkOut()
    xout_nn_1.setStreamName("nn_1")

    # Overwrite default input queue behavior from blocking to overwriting
    # Why?
    xout_rgb.input.setBlocking(False)
    xout_nn_1.input.setBlocking(False)

    # This links the passthrough message with the XLink output from the camera
    # Why? Maybe because of the nature of the output from deeplab? Heatmap?
    # N.B. Passthrough message is suitable for when the queue is set to overwriting
    nn_1.passthrough.link(xout_rgb.input)   

    # Send the inference data to the host
    nn_1.out.link(xout_nn_1.input)

# ---------------------------------------------------------------------------- 

    # Define a ANN node with OpenPose input
    nn_2 = pipeline.createNeuralNetwork()
    nn_2.setBlobPath(nn_path_2)

    # Parameters relevant for OpenPose
    nn_2.setNumInferenceThreads(2)
    nn_2.input.setQueueSize(1)
    nn_2.input.setBlocking(False) # Queue behavior when full
    nn_2.setNumInferenceThreads(2) # Threads used by the node to run the inference

    # Define host inputs & outputs
    xin_nn_2 = pipeline.createXLinkIn()
    xin_nn_2.setStreamName("nn_2_in")
    xout_nn_2 = pipeline.createXLinkOut()
    xout_nn_2.setStreamName("nn_2")

    # Define XLink inputs and outputs, relevant for 2 stage inference
    # 2nd NN inputs is acquired from the XLink stream
    xin_nn_2.out.link(nn_2.input)
    # Output of the 2nd NN is also to be sent via XLink stream
    nn_2.out.link(xout_nn_2.input)

    return pipeline

# ----------------------------------------------------------------------------

with dai.Device(create_pipeline()) as device:
    print("Starting pipeline...")
    device.startPipeline()


    # Host side queues, parameters must correspond those from the pipeline inputs 
    q_rgb = device.getOutputQueue(name = "rgb_stream", maxSize = 4, blocking = False)
    q_nn_1 = device.getOutputQueue(name = "nn_1", maxSize = 4, blocking = False)
    q_nn_2 = device.getInputQueue(name = "nn_2_in")

    # Main program loop
    # ----------------------------------------------------------------------------

    # Auxiliary variables relevant for output
    start_time = time.time()
    counter = 0
    fps = 0
    layer_info_printed = False
    layer_1_info_printed = False

    while True:
        # Fetch latest results
        in_rgb = q_rgb.tryGet()
        in_nn_1 = q_nn_1.tryGet()

        # RGB camera input (1D array) conversion into Height-Width-Channels (HWC) form
        if in_rgb is not None:
            shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
            frame = in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
            frame = np.ascontiguousarray(frame)

        # ANN results (1D array, fixed size, no matter how much results ANN has produced, results end with -1, the rest is filled with 0s) transformations 
        if in_nn_1 is not None:
            layers = in_nn_1.getAllLayers()
       
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
            layer_1 = in_nn_1.getLayerInt32(layers[0].name)

            # Create numpy output
            dims = layer.dims[::-1]
            layer_1 = np.asarray(layer_1, dtype=np.int32).reshape(dims)

            # Block of code to print an array into a text file
            layer_1_to_print = layer_1.reshape(nn_shape_1, nn_shape_1)
            if not layer_1_info_printed:
                layer_1_file = open("layer_1.txt", "w")
                layer_1_file.write("Layer 1 shape:" + str(np.shape(layer_1_to_print)))
                layer_1_file.write("\n")
                for i in range(np.shape(layer_1_to_print)[0]):
                    for j in range(np.shape(layer_1_to_print)[1]):
                        layer_1_file.write(str(int(layer_1_to_print[i, j])))
                    layer_1_file.write("\n")
                layer_1_file.close()
                layer_1_info_printed = True
            
            # Prepare the deeplabv3 colored blob output
            output_colors = decode_deeplabv3p(layer_1)

            nn2_data = dai.NNData()
            nn2_data.setLayer("0", to_planar(frame, (nn_shape_2_x, nn_shape_2_y)))
            q_nn_2.send(nn2_data)
            
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