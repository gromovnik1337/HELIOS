import numpy as np # numpy - manipulate the packet data returned by depthai
import cv2 # opencv - display the video stream
import depthai as dai # access the camera and its data packets

# Create a pipeline object 
# Inside, nodes and connections are defined 
pipeline = dai.Pipeline()

# Definition of a pipeline
# ----------------------------------------------------------------------------

# Create ColorCamera object, set basic parameters that fit mobilenet-ssd input
# https://docs.luxonis.com/projects/api/en/latest/references/python/#depthai.ColorCamera
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setInterleaved(False)

# Define a ANN node with mobilenet-ssd input
# This node runs a neural inference based on the the input data = blob
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath("./mobilenet.blob")

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
    if in_nn is not None:
        bboxes = np.array(in_nn.getFirstLayerFp16()) # From depthai.NNData, convenience function, returns float values from the first layers FP16 tensor
        bboxes = bboxes[:np.where(bboxes == -1)[0][0]] # This crops and array until the -1 value
        # Single ANN result consists of 7 values
        # After this reshape, each row is a ANN result with 7 columns
        # Row number = result number
        # id, label, confidence, x_min, y_min, x_max, y_max
        bboxes = bboxes.reshape((bboxes.size // 7, 7))
        # Only keep bounding box information (last 4 results) with confidence above 0.8
        bboxes = bboxes[bboxes[:, 2] > 0.8][:, 3:7]

    # Display the results
    if frame is not None:
        for raw_bbox in bboxes:
            bbox = frame_norm(frame, raw_bbox)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.imshow("preview", frame)

    if cv2.waitKey(1) == ord('q'):
        break