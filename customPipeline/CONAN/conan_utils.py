#!/usr/bin/env python3

import numpy as np
import cv2

# Path of the deeplabv3 model
nn_path_1 = "./models/deeplab_v3_plus_mvn2_decoder_513_openvino_2021.2_6shave.blob"
nn_shape_1 = 513

# Path of the OpenPose model
nn_path_2 = "./models/human-pose-estimation-0001_openvino_2021.2_6shave.blob"
nn_shape_2_x = 456
nn_shape_2_y = 256

def decode_deeplabv3p(output_tensor):
    class_colors = [[0,0,0],  [0,255,0]]
    class_colors = np.asarray(class_colors, dtype=np.uint8)
    
    # 1-D output of the layer_1 has to be re-shaped to have the same shape as the input to ANN
    # In the simple case 256 px x 256 px
    output = output_tensor.reshape(nn_shape_1, nn_shape_1)
    # Take only the indices of the full color array that contain the person
    output_colors = np.take(class_colors, output, axis=0)
    return output_colors

def show_deeplabv3p(output_colors, frame):
    # Adds two images together but each is assigned different weight
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_image_arithmetics/py_image_arithmetics.html
    # This will superimpose color "person blob" over the frame
    return cv2.addWeighted(frame,1, output_colors,0.2,0)

# Source: https://github.com/luxonis/depthai-experiments/tree/master/gen2-gaze-estimation
# Converts a rgb camera frame that has to be in numpy array format, dimensions (H X W X 3) into a 
# flat list that can be feed into .NNData() 
def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]    
