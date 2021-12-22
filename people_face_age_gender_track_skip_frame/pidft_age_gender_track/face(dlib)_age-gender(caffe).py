##for video------ removing Duplicate NMS

from centroidtracker import CentroidTracker
from trackableobject import TrackableObject
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import datetime
import time, csv
import numpy as np
import argparse, imutils
import time, dlib, cv2, datetime
from itertools import zip_longest
import tensorflow as tf
from imutils import face_utils

ageProto=r"E:\softweb\Pretrained_model\face_age_gender_model\age_deploy.prototxt"
ageModel=r"E:\softweb\Pretrained_model\face_age_gender_model\age_net.caffemodel"
genderProto=r"E:\softweb\Pretrained_model\face_age_gender_model\gender_deploy.prototxt"
genderModel=r"E:\softweb\Pretrained_model\face_age_gender_model\gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

ageNet=cv2.dnn.readNet(ageProto, ageModel)
genderNet=cv2.dnn.readNet(genderProto, genderModel)


path1 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\face-demographics-walking.mp4"
path2 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\face-demographics-walking-and-pause.mp4"
path3 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\happy_customer.mp4"
path4 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\example_01.mp4"
path5 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\Koala Visits Pharmacy __ ViralHog.mp4"
path6 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\Person Data Analytics_ Mask, Age, Gender, Emotion Detection.mp4"
path7 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\Retail Store Customer Analytics_ Footfall, Count, Age, Gender.mp4"


cap = cv2.VideoCapture(path3)

#FPS
fps_start_time = datetime.datetime.now()
fps = 0
total_frames = 0
writer =None

while True:
    
    ret, frame = cap.read()
    
    if not ret:
        break
    
    #img =imutils.resize(img, width=800)
    total_frames = total_frames + 1
    
    #face detaction
  
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 0)

    new_rects = []

    for rect in rects:
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        if y<0:
            print("a")
            continue
        new_rects.append((x, y, x + w, y + h))

        face_img = frame[y:y+h, x:x+w].copy()

        blob=cv2.dnn.blobFromImage(face_img, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        overlay_text = "%s, %s" % (gender, age)
        cv2.putText(frame, overlay_text ,(x,y), font, 1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h),(0, 255, 0), 2)

    #FPS
    fps_end_time = datetime.datetime.now()
    time_diff = fps_end_time - fps_start_time
    if time_diff.seconds == 0:
        fps = 0.0
    else:
        fps = (total_frames / time_diff.seconds)

    fps_text = "FPS: {:.2f}".format(fps)

    cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
    
    #show images
    cv2.imshow("frame", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
