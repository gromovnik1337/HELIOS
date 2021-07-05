import numpy as np # numpy - manipulate the packet data returned by depthai
import cv2 # opencv - display the video stream
import depthai as dai # access the camera and its data packets
import os 

# Change directory to the file's directory
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Create a pipeline object 
# Inside, nodes and connections are defined 
pipeline = dai.Pipeline()

# Definition of a pipeline
# ----------------------------------------------------------------------------

# Create ColorCamera object, set basic parameters that fit mobilenet-ssd input
# https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.ColorCamera
cam_rgb = pipeline.createColorCamera()
img_width = 192
img_height = 192 
cam_rgb.setPreviewSize(img_width, img_height)
cam_rgb.setInterleaved(False)


# Define a ANN node with mobilenet-ssd input
# This node runs a neural inference based on the the input data = blob
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath('/home/ant18/.cache/blobconverter/movenet_singlepose_lightning_FP16_openvino_2021.3_10shave.blob')

# Connect the camera output to the ANN
# preview is the attribute of the depthai.ColorCamera
# It outputs ImgFrame message that carries BGR/RGB planar/interleaved encoded frame data.
# Here, the preview output of the camera is linked to the input to the ANN so inference can be made
cam_rgb.preview.link(detection_nn.input) 

# Camera frames and ANN results are processed on the camera! To get them to the host machine
# XLink has to be used. In this case, one needs XLinkOut node
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb") # Naming the stream
cam_rgb.preview.link(xout_rgb.input) # This sends frames to host

# Same logic for the inference data 
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# Initialization and start-up of the device
# ----------------------------------------------------------------------------
device = dai.Device(pipeline)
# After this, pipeline is being run on the device and it is sending data via XLink
device.startPipeline()

# Host side output queues are to be defined next, with the stream names that had been assigned earlier
q_rgb = device.getOutputQueue("rgb")
q_nn = device.getOutputQueue("nn")



# Defining necessary structures to consume the data from the queues
frame = None
bboxes = []


# Since the bboxes returned by nn have values from <0..1> range relative to the frame widht/height, 
# this functions is converting the inference results into actual px position
def frame_norm(frame, bbox):
    return (np.array(bbox) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)



class Body:
    def __init__(self, scores=None, keypoints_norm=None):
        self.scores = scores # scores of the keypoints
        self.keypoints_norm = keypoints_norm # Keypoints normalized ([0,1]) coordinates (x,y) in the squared input image
        self.keypoints = None # keypoints coordinates (x,y) in pixels in the source image

    def print(self):
        attrs = vars(self)
        print('\n'.join("%s: %s" % item for item in attrs.items()))

# Main program loop
# ----------------------------------------------------------------------------

while True:
    # Fetch latest results
    in_rgb = q_rgb.tryGet()
    in_nn = q_nn.tryGet()

    # RGB camera input (1D array) conversion into Height Width Channels (HWC) form
    if in_rgb is not None:
        shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
        frame = in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
        frame = np.ascontiguousarray(frame)

    
    # ANN results (1D array, fixed size, no matter how much results ANN has produced, results end with -1, the rest is filled with 0s) transformations 
    keypoints = None
    if in_nn is not None:
        layers = np.array(in_nn.getAllLayers()) # From depthai.NNData, convenience function, returns float values from the first layers FP16 tensor

        keypoints = np.array(in_nn.getLayerFp16(layers[2].name)) 
        keypoints = np.reshape(keypoints,(17,3)) 
    
    # Display the results
    if frame is not None and keypoints is not None:

        colors = [[0, 100, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 255], [0, 100, 255], [0, 255, 0],
          [255, 200, 100], [255, 0, 255], [0, 255, 0], [255, 200, 100], [255, 0, 255], [0, 0, 255], [255, 0, 0],
          [200, 200, 0], [255, 0, 0], [200, 200, 0], [0, 0, 0]]



        

        
        num_of_kpts = 17
        confidence_th = 0.3
                    
        body = Body(scores=keypoints[:,2], keypoints_norm=keypoints[:,[1,0]])
        body.keypoints = (np.array([0, 0]) + body.keypoints_norm * img_width).astype(np.int)

        print( body.keypoints )
        score_thresh = 0.3
        for i,x_y in enumerate(body.keypoints):
            if body.scores[i] > score_thresh:
                if i % 2 == 1:
                    color = (0,255,0) 
                elif i == 0:
                    color = (0,255,255)
                else:
                    color = (0,0,255)
                cv2.circle(frame, (x_y[0], x_y[1]), 4, colors[i], -11)

        cv2.imshow("preview", frame)

    if cv2.waitKey(1) == ord('q'):
        break