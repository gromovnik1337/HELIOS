#!/usr/bin/env python3

from numpy.core.numeric import True_
from conan_utils_pose import *
from conan_utils_seg import *
from conan_utils_gen import *
from keypoint_data_extraction import getKeypointsData

import depthai as dai
import numpy as np
from imutils.video import FPS
import threading
import time
import cv2
import argparse
import skimage
from skimage import measure

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-cam', '--camera', action="store_true", help="Use DepthAI 4K RGB camera for inference (conflicts with -vid)")
parser.add_argument('-vid', '--video', type=str, help="Path to video file to be used for inference (conflicts with -cam)")
args = parser.parse_args()

if not args.camera and not args.video:
    raise RuntimeError("No source selected. Please use either \"-cam\" to use RGB camera as a source or \"-vid <path>\" to run on video")

# Delay correction for the video input
delayCorrection = 4.5

# 02.05.2021 added back to the main script. args value was not transmitted to the utils script
class FPSHandler:
    """Class that handles the FPS counting and the frame delay in case of the video source.
    """
    def __init__(self, cap=None):
        self.timestamp = time.time()
        self.start = time.time()
        self.framerate = cap.get(cv2.CAP_PROP_FPS) if cap is not None else None
        self.frame_cnt = 0
        self.ticks = {}
        self.ticks_cnt = {}

    def next_iter(self):
        if not args.camera:
            frame_delay = 1.0 / self.framerate
            delay = (self.timestamp + frame_delay) - time.time()
            if delay > 0:
                time.sleep(delay*delayCorrection)
        self.timestamp = time.time()
        self.frame_cnt += 1

    def tick(self, name):
        if name in self.ticks:
            self.ticks_cnt[name] += 1
        else:
            self.ticks[name] = time.time()
            self.ticks_cnt[name] = 0

    def tick_fps(self, name):
        if name in self.ticks:
            return self.ticks_cnt[name] / (time.time() - self.ticks[name])
        else:
            return 0

    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)

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

# Push up counting parameters
PUSHUP_COUNTER_ON = True

n=0
Uperpoint = None # Initial uper point value
Lowerpoint = None # Initial lower point value
PushupStance = None # Initial pushapstance value
CountingNoKeypoint = 0 
Tc0 = 0 # Initializing the trace of center of torso height
Tc1 = 0
Tc2 = 0
Lowering = None # Initializing the rising/lowering
Pushup = 0

rwristv = 0
lwristv = 0
ranklev = 0 
lanklev = 0
centoru = 0
centorv = 0
headu = 0
ranklev = 0
lshoulv = 0
relbv = 0
lelbv = 0
thetae = 0
thetah = 0

# ROI parameters
ROI_on = True

prev_x_bb, prev_y_bb, prev_w_bb, prev_h_bb, prev_ph, prev_pw= 10, 10, 10, 10 ,10, 10
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

    # Determine source -relevant parameters
    if args.camera:
        fps = FPSHandler()
    else:
        cap = cv2.VideoCapture(str(Path(args.video).resolve().absolute()))
        fps = FPSHandler(cap)

    start_time = time.time()
    curr_time = time.time()
    stop_deeplab = 10 # Lock-on time
    deeplab_on = False # Needed because the lock-on time can still be ticking but the inference data still hasn't come

    # It takes time for the camera to start or to receive video frames
    # No post data processing is possible until first frame is received
    frame = None

    while True:

        if args.video:
            cap.isOpened()
            cap.read()
            read_correctly, frame = cap.read() # read_correctly currently not used
            dim = (nn_shape_1, nn_shape_1)
            frame = cv2.resize(frame, dim)
            frame = np.ascontiguousarray(frame)
        else:
            # Fetch latest results
            in_rgb = q_rgb.tryGet()
            read_correctly = True

            # RGB camera input (1D array) conversion into Height-Width-Channels (HWC) form
            if in_rgb is not None:
                shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
                frame = in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
                frame = np.ascontiguousarray(frame)

        if frame is not None:    
            passed_time = int(curr_time - start_time)
           #print("Time passed: ", passed_time)

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
                if ROI_on:
                    frame_seg = frame.copy()
                    frame_seg_vis = frame.copy()
                    frame_seg_vis_gray = cv2.cvtColor(frame_seg_vis, cv2.COLOR_BGR2GRAY)
                    frame_seg_human = segment_human(layer_1)                  
                    frame_seg_human = cv2.bitwise_and(frame_seg, frame_seg, mask = frame_seg_human)
                    frame_seg_human_gray = cv2.cvtColor(frame_seg_human, cv2.COLOR_BGR2GRAY)
                    # prev_x_bb, prev_y_bb, prev_w_bb, prev_h_bb = x_bb,y_bb, w_bb, h_bb
                    lb_mk_thres_1= cv2.threshold(frame_seg_human_gray,1,255,cv2.THRESH_BINARY)[1]
                    lb_mk_eroded = cv2.erode(lb_mk_thres_1, None, iterations = 3)
                    lb_mk_dilated = cv2.dilate(lb_mk_eroded,None, iterations = 3)
                    # cv2.imshow("thres",lb_mk_dilated)
                    frame_seg_human_labeled = measure.label(lb_mk_dilated)
                    regions = measure.regionprops(frame_seg_human_labeled)
                    regions.sort(key=lambda x: x.area, reverse=True)
                    if len(regions) > 1:
                        for rg in regions[1:]:
                            frame_seg_human_labeled[rg.coords[:,0], rg.coords[:,1]] = 0
                    one_blob_mask = frame_seg_human_labeled > 0
                    frame_seg_human_filtered = np.zeros_like(frame_seg_human_gray, dtype=np.uint8)
                    frame_seg_human_filtered[one_blob_mask] = frame_seg_human_gray[one_blob_mask]
                    # cv2.imshow("MASK",frame_seg_human_filtered)
                    ret, lb_mk_thres_2= cv2.threshold(frame_seg_human_filtered,12,255,cv2.THRESH_BINARY)

                    frame_seg_human_filtered_contours= cv2.findContours(lb_mk_thres_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                    if len(frame_seg_human_filtered_contours) > 0:
                        x_bb, y_bb, w_bb, h_bb = cv2.boundingRect(frame_seg_human_filtered_contours[0])
                    if w_bb < 50 or h_bb < 50:
                        x_bb, y_bb, w_bb, h_bb = prev_x_bb, prev_y_bb, prev_w_bb, prev_h_bb
                    bb_mask_template = np.zeros_like(frame_seg_vis_gray, dtype=np.uint8)
                    cv2.imshow("ROI",frame_seg_vis_gray[y_bb:y_bb+h_bb,x_bb:x_bb+w_bb])
                    cv2.imshow("feed",frame_seg_human)
                    bb_mask_template[y_bb:y_bb+h_bb,x_bb:x_bb+w_bb] = frame_seg_vis_gray[y_bb:y_bb+h_bb,x_bb:x_bb+w_bb]
                    bb_mask= cv2.threshold(bb_mask_template,1,255,cv2.THRESH_BINARY)[1]
                    bb_mask_bool = bb_mask > 10
                    frame_seg_bb = np.zeros_like(frame_seg_vis, dtype=np.uint8)
                    pw = int(w_bb//10)
                    ph = int(h_bb//10)
                    print("now : ", w_bb, h_bb)
                    print("prev: ", prev_w_bb, prev_h_bb)
                    # frame_seg_bb[y_bb-ph:y_bb+h_bb+ph,x_bb-pw:x_bb+w_bb+pw] = frame_seg_vis[y_bb-ph:y_bb+h_bb+ph,x_bb-pw:x_bb+w_bb+pw]
                    if w_bb*h_bb > prev_w_bb*prev_h_bb:
                        frame_seg_bb[y_bb-ph:y_bb+h_bb+ph,x_bb-pw:x_bb+w_bb+pw] = frame_seg_vis[y_bb-ph:y_bb+h_bb+ph,x_bb-pw:x_bb+w_bb+pw]
                        prev_x_bb, prev_y_bb, prev_w_bb, prev_h_bb, prev_ph, prev_pw = x_bb,y_bb, w_bb, h_bb, ph, pw
                    else:
                        frame_seg_bb[prev_y_bb-prev_ph:prev_y_bb+prev_h_bb+prev_ph,prev_x_bb-prev_pw:prev_x_bb+prev_w_bb+prev_pw] = frame_seg_vis[prev_y_bb-prev_ph:prev_y_bb+prev_h_bb+prev_ph,prev_x_bb-prev_pw:prev_x_bb+prev_w_bb+prev_pw]
                    # print(np.shape(frame_seg_bb))
                    
                    cv2.imshow("Segmented human", frame_seg_bb)
                    
                    frame_seg_human=frame_seg_bb

                else:
                    # Segmented human code without ROI commented out, Stipe & Vice, 06.5.2021

                    # frame_seg = frame.copy()
                    frame_seg_human = frame.copy()
                    # frame_seg_human = segment_human(layer_1)                  
                    # frame_seg_human = cv2.bitwise_and(frame_seg, frame_seg, mask = frame_seg_human)
                    # print(np.shape(frame_seg_human))
                    # cv2.imshow("Segmented human", frame_seg_human)

                # Feed the OpenPose
                nn2_frame_data.setLayer("0", to_planar(frame_seg_human, (nn_shape_2_x, nn_shape_2_y)))
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
                    
                    if PUSHUP_COUNTER_ON == True:

                        # Keypoints of right wrist
                        if k_data['K_4'] == []:
                            rwristv = rwristv
                        else:
                            rwristv = k_data['K_4'][1]

                        # Keypoints of left wrist
                        if k_data['K_7'] == []:
                            lwristv = lwristv
                        else:
                            lwristv = k_data['K_7'][1]

                        # Keypoints of right ankle
                        if k_data['K_10'] == []:    
                            ranklev = ranklev
                        else:
                            ranklev = k_data['K_10'][1]

                        # Keypoints of left ankle
                        if k_data['K_13'] == []:
                            lanklev = lanklev
                        else:
                            lanklev = k_data['K_13'][1]

                        # Keypoints of center of torso
                        if k_data['K_1'] == []:
                            centoru = centoru
                            centorv = centorv
                        else:
                            centoru = k_data['K_1'][0]
                            centorv = k_data['K_1'][1]

                        # Keypoints of head
                        if k_data['K_0'] == []:
                            headu = headu
                        else:
                            headu = k_data['K_0'][0]
                        
                        # Keypoints of right shoulder
                        if k_data['K_2'] == []:
                            ranklev = ranklev
                        else:
                            rshoulv = k_data['K_2'][1]

                        # Keypoints of left shoulder
                        if k_data['K_5'] == []:
                            lshoulv = lshoulv
                        else:
                            lshoulv = k_data['K_5'][1]
                        
                        # Keypoints of right elbow
                        if k_data['K_3'] == []:
                            relbv = relbv
                        else:
                            relbv = k_data['K_3'][1]

                        # Keypoints of left elbow
                        if k_data['K_6'] == []:
                            lelbv = lelbv
                        else:
                            lelbv = k_data['K_6'][1]

                        #determining the position of the person
                        #if headu<centoru: #person has legs on the right side and head on the left - for now not working
                        wristv = lwristv
                        anklev = lanklev
                        shoulv = lshoulv
                        elbv = lelbv
                        thetae = k_data['theta_le']
                        thetah = k_data['theta_lh']
                        #elif headu > centoru: #person has legs on the left side and head on the right - for now not working
                        #    wristv = rwristv
                        #    anklev = ranklev
                        #    shoulv = rshoulv
                        #    elbv = relbv
                        #    thetae = k_data['theta_re']
                        #    thetah = k_data['theta_rh']
                        
                        #Entry in upper Push-up stance - The entry point in the upper position is that the wrist needs to be equal or below the ankles
                        if wristv <= anklev:
                            PushupStance = True
                            CountingNoKeypoint = 0
                        else:
                            CountingNoKeypoint = CountingNoKeypoint+1

                        #Since we are sometimes loosing the points of interest this is a buffer zone that the Pushupstance doesnt automaticly go in False 
                        #if CountingNoKeypoint > 10: - for now not working
                        #    PushupStance = False


                        if PushupStance == True and not ( thetah > 170 and thetah < 190 ):    #checking the legs and back - for now not working            
                            if thetae < 190 and thetae > 170: #checking the hands if they are straight
                                Uperpoint = True
                            else:
                                Uperpoint = False
                            
                            if thetae < 120: #shoulder height < elbow height  - for now not working
                                Lowerpoint = True
                            else:
                                Lowerpoint = False
                        else:
                            print('Your back is bent, please correct it.') 

                        #Tc2 = Tc1 and Tc1 < Tc2 and Tc2 < Tc1 c0 < Tc1 and Tc1 < Tc0 and - for now not working
                        #Tc1 = Tc0
                        #Tc0 = shoulv
                        #print(k_data) - for now not working
                        
                        if Lowerpoint == True:
                            Lowering = True
                        elif Uperpoint == True and Lowering == True:
                            Pushup = Pushup + 1
                            print('Number of Push-ups:', Pushup)
                            Lowering=False
                    
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

                    cv2.imshow("CONAN", frame_with_deeplab)
                else:
                    cv2.imshow("CONAN", frame_display)
    
            curr_time = time.time()
            #print("FPS: {:.2f}".format(fps.fps()))
            
            if cv2.waitKey(1) == ord('q'):
                break

if not args.camera:
    cap.release()            
