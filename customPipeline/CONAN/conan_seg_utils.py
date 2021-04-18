#!/usr/bin/env python3
import numpy as np
import cv2

# Path of the deeplabv3 model
nn_path_1 = "./models/deeplab_v3_plus_mvn2_decoder_513_openvino_2021.2_6shave.blob"
nn_shape_1 = 513
           
def decode_deeplabv3p(output_tensor):
    class_colors = [[0,0,0],  [0,255,0]]
    class_colors = np.asarray(class_colors, dtype=np.uint8)
    
    # 1-D output of the layer_1 re-shaping in the shape of the NN input, to be ready for the display
    output = output_tensor.reshape(nn_shape_1, nn_shape_1)
    # Take only the indices of the full color array that contain the person
    output_colors = np.take(class_colors, output, axis=0)
    return output_colors

def show_deeplabv3p(output_colors, frame):
    # Adds two images together but each one gets assigned different weight
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_image_arithmetics/py_image_arithmetics.html
    # This will superimpose colored "person blob" over the frame
    return cv2.addWeighted(frame, 1, output_colors, 0.2, 0)

def printDeeplabv3Info(layers):
    for layer_nr, layer in enumerate(layers):
                    print(f"Layer {layer_nr}")
                    print(f"Name: {layer.name}")
                    print(f"Order: {layer.order}")
                    print(f"dataType: {layer.dataType}")
                    dims = layer.dims[::-1] # reverse dimensions
                    print(f"dims: {dims}")

def printLayer1Info(layer_1):
    layer_1_to_print = layer_1.reshape(nn_shape_1, nn_shape_1)
    layer_1_file = open("layer_1.txt", "w")
    layer_1_file.write("Layer 1 shape:" + str(np.shape(layer_1_to_print)))
    layer_1_file.write("\n")
    for i in range(np.shape(layer_1_to_print)[0]):
        for j in range(np.shape(layer_1_to_print)[1]):
            layer_1_file.write(str(int(layer_1_to_print[i, j])))
        layer_1_file.write("\n")
    layer_1_file.close()



