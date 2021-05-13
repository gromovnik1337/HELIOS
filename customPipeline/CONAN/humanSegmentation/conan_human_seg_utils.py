#!/usr/bin/env python3

import numpy as np
import cv2

nn_path = "./models/deeplab_v3_plus_mvn2_decoder_256_openvino_2021.2_6shave.blob"
#nn_path = "./models/deeplab_v3_plus_mvn2_decoder_513_openvino_2021.2_6shave.blob"
# nn_path = "./models/deeplab_v3_plus_mvn3_decoder_256_openvino_2021.2_6shave.blob"

nn_shape = 256
if '513' in nn_path:
    nn_shape = 513

def decode_deeplabv3p(output_tensor):
    class_colors = [[0,0,0],  [0,255,0]]
    class_colors = np.asarray(class_colors, dtype=np.uint8)
    
    # 1-D output of the layer_1 has to be re-shaped to have the same shape as the input to ANN
    # In the simple case 256 px x 256 px
    output = output_tensor.reshape(nn_shape, nn_shape)
    # Take only the indices of the full color array that contain the person
    output_colors = np.take(class_colors, output, axis=0)
    return output_colors

def show_deeplabv3p(output_colors, frame):
    # Adds two images together but each is assigned different weight
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_image_arithmetics/py_image_arithmetics.html
    # This will superimpose color "person blob" over the frame
    return cv2.addWeighted(frame,1, output_colors,0.2,0)
