import cv2
import math
import argparse
import datetime
from centroidtracker import CentroidTracker

tracker = CentroidTracker(maxDisappeared=40, maxDistance=50)

#face age and gender
faceProto=r"E:\softweb\Pretrained_model\object_detectiom_model\deploy.prototxt.txt"
faceModel=r"E:\softweb\Pretrained_model\object_detectiom_model\res10_300x300_ssd_iter_140000.caffemodel"


faceNet=cv2.dnn.readNetFromCaffe(faceProto,faceModel)


path1 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\face-demographics-walking.mp4"
path2 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\face-demographics-walking-and-pause.mp4"
path3 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\happy_customer.mp4"
path4 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\example_01.mp4"

cap=cv2.VideoCapture(path2)

padding=20
conf_threshold=0.7

fps_start_time = datetime.datetime.now()
fps = 0
total_frames = 0


#save video
fourcc_codec = cv2.VideoWriter_fourcc(*'XVID')
fps = 10.0
capture_size = (int(cap.get(3)), int(cap.get(4)))

out_ = cv2.VideoWriter("out_face_track_caffe_v2.mp4", fourcc_codec, fps, capture_size)

while True:
    ret,frame=cap.read()
    total_frames = total_frames + 1

    if not ret:
        break
    
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    faceNet.setInput(blob)
    detections=faceNet.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            #cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)


    objects = tracker.update(faceBoxes)
    for (objectId, bbox) in objects.items():
        x11, y11, x22, y22 = bbox
        x11 = int(x11)
        y11 = int(y11)
        x22 = int(x22)
        y22 = int(y22)
            
        #frame time ----end--------


        cv2.rectangle(frameOpencvDnn, (x11, y11), (x22, y22), (0, 0, 255), 2)
        text = "ID: {}".format(objectId)
        #text = "id:{},ft:{}sec".format(objectId, int(dwell_time[objectId]))
        cv2.putText(frameOpencvDnn, text, (x11, y11-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    

    #fps
    
    fps_end_time = datetime.datetime.now()
    time_diff = fps_end_time - fps_start_time
    if time_diff.seconds == 0:
        fps = 0.0
    else:
        fps = (total_frames / time_diff.seconds)

    fps_text = "FPS: {:.2f}".format(fps)

    cv2.putText(frameOpencvDnn, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

    out_.write(frameOpencvDnn)
    cv2.imshow("Application", frameOpencvDnn)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
out_.release()
cv2.destroyAllWindows()
