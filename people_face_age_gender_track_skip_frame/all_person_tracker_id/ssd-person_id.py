##for video------ removing Duplicate NMS

import cv2
import imutils
import numpy as np

import cv2
import datetime
import imutils
import numpy as np

import time
from centroidtracker import CentroidTracker
thres = 0.5

#centroid
#tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)
tracker = CentroidTracker(maxDisappeared=80, maxDistance=80)



#path1 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\test_video.mp4"

path2 = r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\VPD officers target violent shoplifters in month-long initiative - Vancouver Sun.mp4"
path3 = r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\VPD_part1.mp4"
path4= r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\VPD_part2.mp4"
path33= r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\VPD_part11.mp4"
path5=r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\motor_bike.mp4"


cap = cv2.VideoCapture(path5)

cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)

fileNames =r"E:\softweb\Pretrained_model\object_detectiom_model\coco.names"

with open(fileNames, 'r') as names:
    classNames = names.read().split("\n")
    
#img_pth = "data\living_room.jpg" 
configPath =r"E:\softweb\Pretrained_model\object_detectiom_model\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = r"E:\softweb\Pretrained_model\object_detectiom_model\frozen_inference_graph.pb"


net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

#FPS
fps_start_time = datetime.datetime.now()
fps = 0
total_frames = 0
writer =None
while True:
    
    ret, img = cap.read()
    
    if ret:
        total_frames = total_frames + 1
    
        classIds, confs, bbox = net.detect(img, confThreshold=thres)
        
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1,-1)[0])
        confs = list(map(float,confs))
        #print(type(confs[0]))
        #print(confs)
        nms_indices =cv2.dnn.NMSBoxes(bbox, confs, thres,nms_threshold =0.2)
        #print(nms_indices)
        rects =[]
        if len(classIds) != 0:
            for i in nms_indices:
                i =i[0]
                #print(i)
                nms_bbox = bbox[i]
                nms_classIds = classIds[i][0]
                nms_confs = confs[i]
                #print(nms_confs)
                #print(nms_clsIds)
                #print(classNames[i])
                #print(nms_box)
                if classNames[nms_classIds-1]!="person":
                    continue
                Label = '{:0.2f}'.format(float(nms_confs))
                Label = "{}%".format(float(Label)*100)
                label = "{} :{}".format(classNames[nms_classIds-1], Label)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1 )
                #(left, top) and (right, bottom)
                #left, top, right, bottom = boxes
                #cv2.rectangle(img, boxes, (0,255,0), 2)
                x, y, w, h = nms_bbox
                #top = max(y, labelSize[1])
                #cv2.rectangle(img, (x,y),(x+w, y+h) , (0,255,0), 2)
                #cv2.rectangle(img, (x, top - labelSize[1]), (x + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
                #cv2.putText(img,label, (x,top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
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
    
            cv2.rectangle(img, (x1, y1), (x1+x2, y1+y2), (0, 0, 255), 2)
            text = "ID:{}".format(objectId)
            cv2.putText(img, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    
                
        #FPS
        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)
    
        fps_text = "FPS: {:.2f}".format(fps)
    
        cv2.putText(img, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)
        
        #show images
        cv2.imshow("img", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("get image failed")
        break

cap.release()
cv2.destroyAllWindows()
