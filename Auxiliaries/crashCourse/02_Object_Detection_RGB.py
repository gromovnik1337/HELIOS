from pathlib import Path
import cv2      # opencv - display the video stream
import depthai  # access the camera and its data packets
import json

device = depthai.Device('', False)

config={'streams': ['previewout', 'metaout'],
        'ai': {"blob_file":        str(Path('./mobilenet-ssd-custom/mobilenet-ssd_custom.blob').resolve().absolute()),
               "blob_file_config": str(Path('./mobilenet-ssd-custom/mobilenet-ssd_custom.json').resolve().absolute())},
        }

# Create the pipeline using the 'previewout & metaout' stream, establishing the first connection to the device.
pipeline = device.create_pipeline(config=config)

# Retrieve model class labels from model config file.
model_config_file = config["ai"]["blob_file_config"]
mcf = open(model_config_file)
model_config_dict = json.load(mcf)
labels = model_config_dict["mappings"]["labels"]
print(labels)
# ['background' 'aeroplane' 'bicycle' 'bird' 'boat' 'bottle' 'bus' 'car'
#  'cat' 'chair' 'cow' 'diningtable' 'dog' 'horse' 'motorbike' 'person'
#  'pottedplant' 'sheep' 'sofa' 'train' 'tvmonitor']

if pipeline is None:   
    raise RuntimeError('Pipeline creation failed!')
    
detections = []
    
while True:
    # Retrieve NN packets and data packets from the device.
    # A data packet contains the video frame data.
    nnet_packets, data_packets = pipeline.get_available_nnet_and_data_packets()

    for nnet_packet in nnet_packets:
        # Build a list of depthai
        detections = list(nnet_packet.getDetectedObjects())

    for packet in data_packets:

        if packet.stream_name == 'previewout': 
            meta = packet.getMetadata()
            camera = meta.getCameraName()
            window_name = 'previewout-' + camera                  
            data = packet.getData()
            # change shape (3, 300, 300) -> (300, 300, 3)            
            data0 = data[0, :, :]            
            data1 = data[1, :, :]            
            data2 = data[2, :, :]           
            frame = cv2.merge([data0, data1, data2])
            img_h = frame.shape[0]            
            img_w = frame.shape[1]            

            for detection in detections:
                #detection_dict = detection.get_dict()
                #print(detection_dict)
                # {'label': 5, 'confidence': 0.89453125, 'x_min': 0.42529296875, 'y_min': 0.136962890625, 'x_max': 0.65478515625, 'y_max': 0.8837890625, 'depth_x': 0.0, 'depth_y': 0.0, 'depth_z': 0.0}

                pt1 = int(detection.x_min * img_w), int(detection.y_min * img_h)
                pt2 = int(detection.x_max * img_w), int(detection.y_max * img_h)
                label = labels[int(detection.label)]
                score = int(detection.confidence * 100)
                cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
                cv2.putText(frame, str(score) + ' ' + label,(pt1[0] + 2, pt1[1] + 15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow(window_name, frame)      

    if cv2.waitKey(1) == ord('q'):
        break

# The pipeline object should be deleted after exiting the loop. Otherwise device will continue working.
# This is required if you are going to add code after exiting the loop.
del pipeline                                          
del device