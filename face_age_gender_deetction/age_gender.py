import cv2
import math
import argparse
import datetime
from centroidtracker import CentroidTracker

tracker = CentroidTracker(maxDisappeared=70, maxDistance=80)

#face age and gender
faceProto="deploy.prototxt.txt"
faceModel="res10_300x300_ssd_iter_140000.caffemodel"

ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['M','F']

faceNet=cv2.dnn.readNetFromCaffe(faceProto,faceModel)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)



path2 ="face-demographics-walking-and-pause.mp4"

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

#out_ = cv2.VideoWriter("out_face_track_id_age_gender(all_caffe)_v1.4.mp4", fourcc_codec, fps, capture_size)

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


        cv2.rectangle(frameOpencvDnn, (x11, y11), (x22, y22), (255, 255, 0), 2)
        text = "ID: {}".format(objectId)
        #text = "id:{},ft:{}sec".format(objectId, int(dwell_time[objectId]))
        #cv2.putText(frameOpencvDnn, text, (x11, y11-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    
        #resultImg =frameOpencvDnn.copy()

        #age-gender
        faceBox =bbox
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        cv2.putText(frameOpencvDnn, f'{gender}, {age}', (x11, y11-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 1)
        #cv2.imshow("Detecting age and gender", resultImg)
    
        
    #fps
    
    fps_end_time = datetime.datetime.now()
    time_diff = fps_end_time - fps_start_time
    if time_diff.seconds == 0:
        fps = 0.0
    else:
        fps = (total_frames / time_diff.seconds)

    fps_text = "FPS: {:.2f}".format(fps)

    #cv2.putText(frameOpencvDnn, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

    #write
   # out_.write(frameOpencvDnn)
    
    cv2.imshow("Application", frameOpencvDnn)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
#out_.release()
cv2.destroyAllWindows()
