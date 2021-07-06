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

# Create ColorCamera object, set basic parameters
# https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.ColorCamera
rgb = pipeline.createColorCamera()
img_width = 192
img_height = 192 
rgb.setPreviewSize(img_width, img_height)
rgb.setInterleaved(False)

# Define a ANN node

nn_2 = pipeline.createNeuralNetwork()
nn_2.setBlobPath('./movenet.blob')

# Camera frames and ANN results are processed on the camera! To get them to the host machine
# XLink has to be used. In this case, one needs XLinkOut node
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb_stream") # Naming the stream

rgb.preview.link(xout_rgb.input) # This sends frames to host

# Define inputs & outputs to the host
xin_nn_2 = pipeline.createXLinkIn() 
xin_nn_2.setStreamName("nn_2_in")

xout_nn_2 = pipeline.createXLinkOut()
xout_nn_2.setStreamName("nn_2_out")

# Input to the 2nd NN is received from the XLink stream
xin_nn_2.out.link(nn_2.input)

# Send the inference data to the host
nn_2.out.link(xout_nn_2.input)


# Initialization and start-up of the device
# ----------------------------------------------------------------------------

# Frame that is to be sent inside XLink to perform inference
nn2_frame_data = dai.NNData()

device = dai.Device(pipeline)
# After this, pipeline is being run on the device and it is sending data via XLink
device.startPipeline()

# Host side output queues are to be defined next, with the stream names that had been assigned earlier
q_rgb = device.getOutputQueue("rgb_stream")
q_nn_2_in = device.getInputQueue(name = "nn_2_in", maxSize = 1, blocking = False)
q_nn_2_out = device.getOutputQueue(name = "nn_2_out", maxSize = 1, blocking = False)



# Defining necessary structures to consume the data from the queues
frame = None
bboxes = []


# Since the bboxes returned by nn have values from <0..1> range relative to the frame widht/height, 
# this functions is converting the inference results into actual px position
def frame_norm(frame, bbox):
    return (np.array(bbox) * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]).astype(int)

# Converts a rgb camera frame that has to be in numpy array format, dimensions (H X W X 3) into a 
# flat list that can be feed into .NNData() 
def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return cv2.resize(arr, shape).transpose(2,0,1).flatten()

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

    # RGB camera input (1D array) conversion into Height Width Channels (HWC) form
    if in_rgb is not None:
        shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
        frame = in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
        frame = np.ascontiguousarray(frame)

        nn2_frame_data.setLayer("0", to_planar(frame, (192, 192)))
        q_nn_2_in.send(nn2_frame_data)

    in_nn_2 = q_nn_2_out.tryGet()

    # ANN results (1D array, fixed size, no matter how much results ANN has produced, results end with -1, the rest is filled with 0s) transformations 
    keypoints = None
    if in_nn_2 is not None:
        layers = np.array(in_nn_2.getAllLayers()) # From depthai.NNData, convenience function, returns float values from the first layers FP16 tensor

        keypoints = np.array(in_nn_2.getLayerFp16(layers[2].name)) 
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

  