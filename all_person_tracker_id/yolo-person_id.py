import cv2
import datetime
import imutils
import numpy as np

import time
from centroidtracker import CentroidTracker


confThreshold =0.5
nmsThreshold= 0.2

#path =r"E:\softweb\ML\project\object_detection-transfer-learning-yolo-ssd-mobilenet\data\motor_bike.mp4"
#path1 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\test_video.mp4"

path2 = r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\VPD officers target violent shoplifters in month-long initiative - Vancouver Sun.mp4"
path3 = r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\VPD_part1.mp4"
path4= r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\VPD_part2.mp4"
path33= r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\VPD_part11.mp4"
path5=r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\motor_bike.mp4"


#image_pth = r"E:\softweb\ML\project\object_detection\data\living_room.jpg"
# load the COCO class labels our YOLO model was trained on
#image_pth =  r"E:\softweb\ML\project\object_detection-transfer-learning-yolo-ssd-mobilenet\data\living_room.jpg" 
# load the COCO class labels our YOLO model was trained on
labelsPath = r'E:\softweb\Pretrained_model\yolo_weights\coco.names'
# derive the paths to the YOLO weights and model configuration
weightsPath = r'E:\softweb\Pretrained_model\yolo_weights\yolov4-tiny.weights'
configPath = r'E:\softweb\Pretrained_model\yolo_weights\yolov4-tiny.cfg'

#centroid
#tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)
tracker = CentroidTracker(maxDisappeared=50, maxDistance=50)


LABELS  =[]
with open(labelsPath, "r") as f:
    LABELS  = f.read().strip("\n").split("\n")
    
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

net =cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#image =cv2.imread(image_pth)
#(H, W) = image.shape[:2]
cap =cv2.VideoCapture(path5)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the input image and then perform a forward
# pass of the YOLO object detector, giving us our bounding boxes and
# associated probabilities

def generate_boxes_confidences_classids(layerOutputs, H, W, confThreshold):
	boxes = []
	confidences = []
	classIDs = []
	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if confidence > confThreshold:
				if LABELS[classID] != "person":
					continue
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
	return boxes, confidences, classIDs

#FPS
fps_start_time = datetime.datetime.now()
fps = 0
total_frames = 0
writer =None
while True:
    ret, image =cap.read()
    #image = imutils.resize(image, width=600)
    total_frames = total_frames + 1
    
    if not ret:
        break
    
    (H, W) = image.shape[:2]
    

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
    	swapRB=True, crop=False)
    
    net.setInput(blob)
    start = time.time()
    
    layerOutputs = net.forward(ln)
    # [,frame,no of detections,[classid,class score,conf,x,y,h,w]
    end = time.time()
    
    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    rects =[]
    boxes, confidences, classIDs = generate_boxes_confidences_classids(layerOutputs, H, W, 0.5)
    
    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    
    # ensure at least one detection exists
    
     # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            rects.append((x, y, w, h))
            

    #tracker
    objects = tracker.update(rects)
    print(objects)
    objectId_ls =[]
    for (objectId, bbox) in objects.items():
        x1, y1, x2, y2 = bbox
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        objectId_ls.append(objectId)

        cv2.rectangle(image, (x1, y1), (x1+x2, y1+y2), (0, 0, 255), 2)
        text = "ID:{}".format(objectId)
        cv2.putText(image, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)


    #FPS
    fps_end_time = datetime.datetime.now()
    time_diff = fps_end_time - fps_start_time
    if time_diff.seconds == 0:
        fps = 0.0
    else:
        fps = (total_frames / time_diff.seconds)

    fps_text = "FPS: {:.2f}".format(fps)

    cv2.putText(image, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)
    # show the output image
    cv2.imshow("Image", image)
    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter("output_personID3.mp4", fourcc, 30,
                                 (image.shape[1], image.shape[0]), True)

    #writer.write(image)
    
print("list of all object id:", objectId_ls)
#writer.release()
cap.release()
cv2.destroyAllWindows()

