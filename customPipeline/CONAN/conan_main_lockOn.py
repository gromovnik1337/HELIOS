#!/usr/bin/env python3

from conan_utils_pose import *
from conan_utils_seg import *
from conan_utils_gen import *
from keypoint_data_extraction import getKeypointsData
# import skimage
# from skimage import measure

import depthai as dai
import numpy as np
from imutils.video import FPS
import threading
import time
import cv2

import threading
from pathlib import Path

# Definition of a pipeline
# ----------------------------------------------------------------------------
def create_pipeline():
    pipeline = dai.Pipeline()
    # Copy pasted from human segmentation file, reason unknown :)
    pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_2)

    # Define camera object
    rgb = pipeline.createColorCamera()

    # Set the resolution
    rgb.setPreviewSize(nn_shape_1, nn_shape_1)
    # rgb.setPreviewSize(1920, 1080)

    rgb.setInterleaved(False)
    rgb.setFps(60)

    # Define outputs to the host
    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb_stream")

    rgb.preview.link(xout_rgb.input)

    # Overwrite default input queue behavior from blocking to overwriting
    # Why?
    xout_rgb.input.setBlocking(False)
    
    # Define a ANN node with deeplabv3_person_513 input
    nn_1 = pipeline.createNeuralNetwork()
    nn_1.setBlobPath(nn_path_1)

    # Parameters relevant for deepblabv3
    nn_1.setNumPoolFrames(4)
    nn_1.input.setBlocking(False) # Queue behavior when full
    nn_1.setNumInferenceThreads(2) # Threads used by the node to run the inference

    # Define inputs & outputs to the host
    xin_nn_1 = pipeline.createXLinkIn() 
    xin_nn_1.setStreamName("nn_1_in")
  
    xout_nn_1 = pipeline.createXLinkOut()
    xout_nn_1.setStreamName("nn_1_out")

    # Input to the 1st NN is received from the XLink stream
    xin_nn_1.out.link(nn_1.input)  

    # Send the inference data to the host
    nn_1.out.link(xout_nn_1.input)

# ---------------------------------------------------------------------------- 

    # Define a ANN node with OpenPose input
    nn_2 = pipeline.createNeuralNetwork()
    nn_2.setBlobPath(nn_path_2)

    # Parameters relevant for OpenPose
    nn_2.input.setQueueSize(1)
    nn_2.input.setBlocking(False) # Queue behavior when full
    nn_2.setNumInferenceThreads(2) # Threads used by the node to run the inference

    # Define inputs & outputs to the host
    xin_nn_2 = pipeline.createXLinkIn() 
    xin_nn_2.setStreamName("nn_2_in")

    xout_nn_2 = pipeline.createXLinkOut()
    xout_nn_2.setStreamName("nn_2_out")

    # Input to the 2nd NN is received from the XLink stream
    xin_nn_2.out.link(nn_2.input)

    # Send the inference data to the host
    nn_2.out.link(xout_nn_2.input)

    return pipeline

# ----------------------------------------------------------------------------

keypoints_list = None
detected_keypoints = None
personwiseKeypoints = None
prev_x_bb, prev_y_bb, prev_w_bb, prev_h_bb = 0, 0, 0, 0
x_bb, y_bb, w_bb, h_bb = 0, 0, 513, 513

# Frame that is to be sent inside XLink to perform inference
nn1_frame_data = dai.NNData()
nn2_frame_data = dai.NNData()

# Create the pipeline object
pipeline = create_pipeline()

with dai.Device(pipeline) as device:
    print("Starting pipeline...")
    device.startPipeline()

    # Host side queues, parameters must correspond those from the pipeline inputs 
    q_rgb = device.getOutputQueue(name = "rgb_stream", maxSize = 4, blocking = False)

    q_nn_1_in = device.getInputQueue(name = "nn_1_in", maxSize = 1, blocking = False)
    q_nn_1_out = device.getOutputQueue(name = "nn_1_out", maxSize = 4, blocking = False)
    
    q_nn_2_in = device.getInputQueue(name = "nn_2_in", maxSize = 1, blocking = False)
    q_nn_2_out = device.getOutputQueue(name = "nn_2_out", maxSize = 1, blocking = False)
    
    # Main program loop
    # ----------------------------------------------------------------------------

    # Auxiliary variables relevant for output
    layer_info_printed = False
    layer_1_info_printed = False

    fps = FPSHandler() # TODO Change when you create video input possibility. Changes recquired inside the FPS handler class

    start_time = time.time()
    curr_time = time.time()
    stop_deeplab = 45 # Lock-on time
    deeplab_on = False # Needed because the lock-on time can still be ticking but the inference data still hasn't come

    # It takes time for the camera to start
    # No post data processing is possible until first frame is received
    frame = None

    while True:
        # Fetch latest results
        in_rgb = q_rgb.tryGet()

        # RGB camera input (1D array) conversion into Height-Width-Channels (HWC) form
        if in_rgb is not None:
            shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
            frame = in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
            frame = np.ascontiguousarray(frame)

        if frame is not None:    
            passed_time = int(curr_time - start_time)
            print("Time passed: ", passed_time)

            if passed_time < stop_deeplab:
                
                # ANN results (1D array, fixed size, no matter how much results ANN has produced, results end with -1, the rest is filled with 0s) transformations 
                nn1_frame_data.setLayer("0", to_planar(frame, (nn_shape_1, nn_shape_1)))
                q_nn_1_in.send(nn1_frame_data)
                in_nn_1 = q_nn_1_out.tryGet()

                if in_nn_1 is not None:
                    deeplab_on = True
                    layers = in_nn_1.getAllLayers()

                    # Print deepplabv3 info
                    if not layer_info_printed:
                        printDeeplabv3Info(layers)
                        layer_info_printed = True
                    # Relevant information is in layer1
                    layer_1 = in_nn_1.getLayerInt32(layers[0].name)

                    # Create numpy output
                    dims = layers[0].dims[::-1] # reverse dimensions
                    layer_1 = np.asarray(layer_1, dtype=np.int32).reshape(dims)

                    # Print the first layer into a .txt file
                    if not layer_1_info_printed:
                        printLayer1Info(layer_1)              
                        layer_1_info_printed = True
                
                    # Take only X and Y (frame of 1s and 0s) part of the deeplab output
                    layer_1_to_lock_on = layer_1.reshape(nn_shape_1, nn_shape_1)
                    # Sum up the rows and the columns, find where is the max = that is the center of the human
                    human_x = np.argmax(np.sum(layer_1_to_lock_on, 0))
                    human_y = np.argmax(np.sum(layer_1_to_lock_on, 1))

                    # Prepare the deeplabv3 colored blob output
                    output_colors = decode_deeplabv3p(layer_1)


            if deeplab_on is True:
                # Segment the human
                # Make a separate frame not to superimpose the results of the first bitwise operation
                frame_seg = frame.copy()
                frame_seg_vis = frame.copy()
                frame_seg_vis_gray = cv2.cvtColor(frame_seg_vis, cv2.COLOR_BGR2GRAY)
                frame_seg_human = segment_human(layer_1)                  
                frame_seg_human = cv2.bitwise_and(frame_seg, frame_seg, mask = frame_seg_human)
                frame_seg_human_gray = cv2.cvtColor(frame_seg_human, cv2.COLOR_BGR2GRAY)
                prev_x_bb, prev_y_bb, prev_w_bb, prev_h_bb = x_bb,y_bb, w_bb, h_bb
                lb_mk_thres_1= cv2.threshold(frame_seg_human_gray,1,255,cv2.THRESH_BINARY)[1]
                lb_mk_eroded = cv2.erode(frame_seg_human_gray, None, iterations = 3)
                lb_mk_dilated = cv2.dilate(lb_mk_eroded,None, iterations = 3)
                # cv2.imshow("lb_mk_thres", lb_mk_dilated)
                segmented_object_filtered_contours= cv2.findContours(lb_mk_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                x_bb, y_bb, w_bb, h_bb = cv2.boundingRect(segmented_object_filtered_contours[0])
                if w_bb < 20 or h_bb < 20:
                    x_bb, y_bb, w_bb, h_bb = prev_x_bb, prev_y_bb, prev_w_bb, prev_h_bb
                bb_mask_template = np.zeros_like(frame_seg_vis_gray, dtype=np.uint8)
                cv2.imshow("ROI",frame_seg_vis_gray[y_bb:y_bb+h_bb,x_bb:x_bb+w_bb])
                bb_mask_template[y_bb:y_bb+h_bb,x_bb:x_bb+w_bb] = frame_seg_vis_gray[y_bb:y_bb+h_bb,x_bb:x_bb+w_bb]
                bb_mask= cv2.threshold(bb_mask_template,0,255,cv2.THRESH_BINARY)[1]
                bb_mask_bool = bb_mask > 10
                frame_seg_bb = np.zeros_like(frame_seg_vis, dtype=np.uint8)
                frame_seg_bb[bb_mask_bool] = frame_seg_vis[bb_mask_bool]
                padding_width = int(w_bb//10)
                padding_height = int(h_bb//10)
                print(np.shape(frame_seg_human))
                print(x_bb, y_bb, w_bb, h_bb)
                cv2.imshow("Segmented human", frame_seg_human)
                # cv2.imshow("Segmented Bounding Box", frame_seg_bb)
                # Feed the OpenPose
                nn2_frame_data.setLayer("0", to_planar(frame_seg_bb, (nn_shape_2_x, nn_shape_2_y)))
                q_nn_2_in.send(nn2_frame_data)

            else:
                # Feed the OpenPose
                nn2_frame_data.setLayer("0", to_planar(frame, (nn_shape_2_x, nn_shape_2_y)))
                q_nn_2_in.send(nn2_frame_data)


            # Start the OpenPose
            in_nn_2 = q_nn_2_out.tryGet()

            if in_nn_2 is not None:

                # Only the FPS performance of OpenPose is measured
                fps.tick('nn')
                
                heatmaps = np.array(in_nn_2.getLayerFp16('Mconv7_stage2_L2')).reshape((1, 19, 32, 57))
                pafs = np.array(in_nn_2.getLayerFp16('Mconv7_stage2_L1')).reshape((1, 38, 32, 57))
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

            h, w = frame.shape[:2] 

            if frame is not None:
                # Create a copy of frame that will be used for plotting (lines, blobs, points etc.)
                # One frame has to be "clean" to be used for segmentation results and feeding NN
                # If the RGB input resolution is different than deeplabs input, show_deeplabv3p function does not work
                # This if statement makes sure that deeplab always get expected frame size
                if deeplab_on is True: 
                    frame_display = frame.copy()
                    frame_display = cv2.resize(frame, (nn_shape_1, nn_shape_1) )
                else:
                    frame_display = frame.copy()
                    pass

                # One round of OpenPose inference fps tracking done    
                fps.next_iter()

                if keypoints_list is not None and detected_keypoints is not None and personwiseKeypoints is not None:
                    #---------------------------------------------------------------
                    # Compute the keypoint data, returns a dictionary with all the variables of interest + keypoints
                    k_data = getKeypointsData(detected_keypoints)
                    
                    # To see what is available in the dictionary
                    # print(k_data.keys())
                    # currently: dict_keys(['K_0', 'K_1', 'K_2', 'K_3', 'K_4', 'K_5', 'K_6', 'K_7', 'K_8', 'K_9', 'K_10', 'K_11', 'K_12', 'K_13', 'K_14', 'K_15', 'K_16', 'K_17', 'd_le_ls', 'd_re_rs', 'd_re_rw', 'd_le_lw', 'd_ls_lw', 'd_rs_rw', 'd_rs_ls', 'theta_le', 'theta_re'])
                    
                    # Accessing a variable:
                    angle = k_data['theta_re']
                    d = k_data['d_re_rs']
                    print('Got angle: ' + str(angle) + ', and distance: ' + str(d))
                    #---------------------------------------------------------------
                    for i in range(18):
                        for j in range(len(detected_keypoints[i])):
                            cv2.circle(frame_display, detected_keypoints[i][j][0:2], 5, colors[i], -1, cv2.LINE_AA)

                    for i in range(17):
                        for n in range(len(personwiseKeypoints)):
                            index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
                            if -1 in index:
                                continue
                            B = np.int32(keypoints_list[index.astype(int), 0])
                            A = np.int32(keypoints_list[index.astype(int), 1])

                            cv2.line(frame_display, (B[0], A[0]), (B[1], A[1]), colors[i], 3, cv2.LINE_AA)
                
                cv2.putText(frame_display, f"RGB FPS: {round(fps.fps(), 1)}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                cv2.putText(frame_display, f"OpenPose FPS:  {round(fps.tick_fps('nn'), 1)}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                cv2.putText(frame_display, "Frame size: {0}x{1}".format(h,w), (340, frame_display.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))

                if deeplab_on is True:
                    # Superimpose the green blob
                    frame_with_deeplab = show_deeplabv3p(output_colors, frame_display)

                    # Create the crosshair
                    cv2.circle(frame_with_deeplab, (human_x, human_y), 15, (0, 0, 255), 2)
                    cv2.rectangle(frame_with_deeplab, (human_x - 20, human_y + 20), (human_x + 20, human_y - 20), (0, 0, 255), 5)
                    cv2.line(frame_with_deeplab, (human_x, human_y + 30), (human_x, human_y - 30), (0, 0, 255), 2)
                    cv2.line(frame_with_deeplab, (human_x - 30, human_y), (human_x + 30, human_y), (0, 0, 255), 2)
                    
                    cv2.rectangle(frame_with_deeplab,(x_bb,y_bb),(x_bb+w_bb,y_bb+h_bb),(0,255,0),2)
                    cv2.rectangle(frame_with_deeplab,(x_bb-padding_width,y_bb-padding_height),(x_bb+w_bb+padding_width,y_bb+h_bb+padding_height),(225,0,0),2)

                    cv2.imshow("CONAN", frame_with_deeplab)
                else:
                    cv2.imshow("CONAN", frame_display)
    
            curr_time = time.time()
            print("FPS: {:.2f}".format(fps.fps()))
            
            if cv2.waitKey(1) == ord('q'):
                break
