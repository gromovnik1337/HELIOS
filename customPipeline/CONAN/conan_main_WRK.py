#!/usr/bin/env python3

from conan_utils import *
import depthai as dai
import numpy as np
from imutils.video import FPS
import threading
import time
import cv2

import threading
from pathlib import Path
from pose import getKeypoints, getValidPairs, getPersonwiseKeypoints

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
    xin_nn_2 = pipeline.createXLinkIn() # Equivalent to pose_in in main.py of the OpenPose
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

colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 0],
          [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 0, 255], [255, 0, 0],
          [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]
POSE_PAIRS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
              [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 17], [5, 16]]

running = True
pose = None
keypoints_list = None
detected_keypoints = None
personwiseKeypoints = None

def pose_thread(in_queue):
    global keypoints_list, detected_keypoints, personwiseKeypoints

    while running:
        try:
            raw_in = in_queue.get()
        except RuntimeError:
            return
        #fps.tick('nn')
        heatmaps = np.array(raw_in.getLayerFp16('Mconv7_stage2_L2')).reshape((1, 19, 32, 57))
        pafs = np.array(raw_in.getLayerFp16('Mconv7_stage2_L1')).reshape((1, 38, 32, 57))
        heatmaps = heatmaps.astype('float32')
        pafs = pafs.astype('float32')
        outputs = np.concatenate((heatmaps, pafs), axis=1)

        new_keypoints = []
        new_keypoints_list = np.zeros((0, 3))
        keypoint_id = 0

        for row in range(18):
            probMap = outputs[0, row, :, :]
            probMap = cv2.resize(probMap, (w, h))  # (456, 256)
            keypoints = getKeypoints(probMap, 0.3)
            new_keypoints_list = np.vstack([new_keypoints_list, *keypoints])
            keypoints_with_id = []

            for i in range(len(keypoints)):
                keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                keypoint_id += 1

            new_keypoints.append(keypoints_with_id)

        valid_pairs, invalid_pairs = getValidPairs(outputs, w, h, new_keypoints)
        newPersonwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs, new_keypoints_list)

        detected_keypoints, keypoints_list, personwiseKeypoints = (new_keypoints, new_keypoints_list, newPersonwiseKeypoints)


with dai.Device(create_pipeline()) as device:
    print("Starting pipeline...")
    device.startPipeline()

    # Host side queues, parameters must correspond those from the pipeline inputs 
    q_rgb = device.getOutputQueue(name = "rgb_stream", maxSize = 4, blocking = False)
    q_nn_1 = device.getOutputQueue(name = "nn_1", maxSize = 4, blocking = False)
    
    q_nn_2 = device.getOutputQueue(name = "nn_2", maxSize = 1, blocking = False)
    q_nn_2_in = device.getInputQueue(name = "nn_2_in")

    def should_run():
        return True


    def get_frame():
        return True, np.array(cam_out.get().getData()).reshape((3, 256, 456)).transpose(1, 2, 0).astype(np.uint8)

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

            # Create the output for the OpenPose
            nn2_data = dai.NNData()
            nn2_data.setLayer("0", to_planar(frame, (nn_shape_2_x, nn_shape_2_y)))
            q_nn_2_in.send(nn2_data)

            t = threading.Thread(target=pose_thread, args=(q_nn_2, ))
            t.start()

            #try:
                #while should_run():
                    #read_correctly, frame = get_frame()

                    #if not read_correctly:
                    #break

            h, w = frame.shape[:2]  # 256, 456
            debug_frame = frame.copy()
     
            if keypoints_list is not None and detected_keypoints is not None and personwiseKeypoints is not None:
                for i in range(18):
                    for j in range(len(detected_keypoints[i])):
                        cv2.circle(debug_frame, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)

              
                for i in range(17):
                    for n in range(len(personwiseKeypoints)):
                        index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                        if -1 in index:
                            continue
                        B = np.int32(keypoints_list[index.astype(int), 0])
                        A = np.int32(keypoints_list[index.astype(int), 1])

                        cv2.line(debug_frame, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)

            #cv2.putText(debug_frame, f"RGB FPS: {round(fps.fps(), 1)}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            #v2.putText(debug_frame, f"NN FPS:  {round(fps.tick_fps('nn'), 1)}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            cv2.imshow("rgb", debug_frame)           
        
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
      
t.join()