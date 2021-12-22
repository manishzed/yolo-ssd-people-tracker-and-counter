# ----------------------------------------------
# Yolo Keras Face Detection from WebCamera
# ----------------------------------------------

from datetime import datetime
import numpy as np
import sys, getopt
import cv2
import os
from tensorflow.keras import backend as K
import datetime
#os.environ['KERAS_BACKEND'] = 'tensorflow'

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import tensorflow as tf

from centroidtracker import CentroidTracker
from collections import defaultdict

tracker = CentroidTracker(maxDisappeared=40, maxDistance=50)

ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']


def get_range(age):
    new_age = ''
    if age>=0 and age<=2:
        print(ageList[0])
        new_age =ageList[0]
    elif age>2 and age<=6:
        print(ageList[1])
        new_age =ageList[1]
    elif age>6 and age<=12:
        print(ageList[2])
        new_age =ageList[2]
    elif age>12 and age<=20:
        print(ageList[3])
        new_age =ageList[3]
    elif age>20 and age<=32:
        print(ageList[4])
        new_age =ageList[4]
    elif age>32 and age<=43:
        print(ageList[5])
        new_age =ageList[5]
    elif age>43 and age<=53:
        print(ageList[6])
        new_age =ageList[6]
    elif age>53 and age<=100:
        print(ageList[7])
        new_age =ageList[7]
    return new_age

#caffe face
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            #cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (255,255,0), int(round(frameHeight/150)), 5)
    return frameOpencvDnn,faceBoxes

#display result
def show_results(resultImg,frame, faceBoxes, model_age, model_gender,out_, padding=20):
    
    objects = tracker.update(faceBoxes)
    for (objectId, bbox) in objects.items():
        x11, y11, x22, y22 = bbox
        x11 = int(x11)
        y11 = int(y11)
        x22 = int(x22)
        y22 = int(y22)

        cv2.rectangle(resultImg, (x11, y11), (x22, y22), (255, 255, 0), 2)
        #text = "ID: {}".format(objectId)
        #text = "id:{},ft:{}sec".format(objectId, int(dwell_time[objectId]))
        #cv2.putText(resultImg, text, (x11, y11-50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    
   
        faceBox = bbox
        face_image = frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]
                
            
        #lines_age=open('words/agegender_age_words.txt').readlines()
        #lines_gender=open('words/agegender_gender_words.txt').readlines()
        
        lines_age =['(0, 2)','(4, 6)','(8, 13)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
        lines_gender = ['F', 'M']
        

        
        if(model_age!=None):
            shape = model_age.layers[0].get_output_at(0).get_shape().as_list()
            print("model_age", model_age)
            print("shape", shape)
            img_keras = cv2.resize(face_image, (shape[1],shape[2]))
            #img_keras = img_keras[::-1, :, ::-1].copy()    #BGR to RGB
            img_keras = np.expand_dims(img_keras, axis=0)
            img_keras = img_keras / 255.0
            
            pred_age_keras = model_age.predict(img_keras)[0]
            print("pred_age_keras", pred_age_keras)
            prob_age_keras = np.max(pred_age_keras)
            cls_age_keras = pred_age_keras.argmax()
    
            age=0
            for i in range(101):
                age=age+pred_age_keras[i]*i
            label_x=str(int(age))
            
            print("age_:", label_x)
            label = get_range(age)
            
            #label="%.2f" % prob_age_keras + " " + lines_age[cls_age_keras]
            #print(cls_age_keras)
    
            cv2.putText(resultImg, label, (x11,y11-30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0,165,255));
    
        if(model_gender!=None):
            shape = model_gender.layers[0].get_output_at(0).get_shape().as_list()
    
            img_gender = cv2.resize(face_image, (shape[1],shape[2]))
            #img_gender = img_gender[::-1, :, ::-1].copy()    #BGR to RGB
            img_gender = np.expand_dims(img_gender, axis=0)
            img_gender = img_gender / 255.0
            
            pred_gender_keras = model_gender.predict(img_gender)[0]
            prob_gender_keras = np.max(pred_gender_keras)
            cls_gender_keras = pred_gender_keras.argmax()
            
            print("gender:",cls_gender_keras)
            cv2.putText(resultImg, lines_gender[cls_gender_keras], (x11,y11-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0,165,255));
        
    #out_.write(resultImg)
    cv2.imshow('YoloKerasFaceDetection',resultImg)

def main():
    #fps
    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0
    
   
    path1 =r"E:\softweb\priyasoftweb\person_tracker_id_ft_age-gender_UBI-IOT\input\face-demographics-walking.mp4"
    path2 =r"E:\softweb\priyasoftweb\person_tracker_id_ft_age-gender_UBI-IOT\input\face-demographics-walking-and-pause.mp4"
    path3 =r"E:\softweb\priyasoftweb\person_tracker_id_ft_age-gender_UBI-IOT\input\happy_customer.mp4"
    path4 =r"E:\softweb\priyasoftweb\person_tracker_id_ft_age-gender_UBI-IOT\input\example_01.mp4"
    path5 =r"E:\softweb\priyasoftweb\person_tracker_id_ft_age-gender_UBI-IOT\input\Koala Visits Pharmacy __ ViralHog.mp4"
    path6 =r"E:\softweb\priyasoftweb\person_tracker_id_ft_age-gender_UBI-IOT\input\Person Data Analytics_ Mask, Age, Gender, Emotion Detection.mp4"
    path7 =r"E:\softweb\priyasoftweb\person_tracker_id_ft_age-gender_UBI-IOT\input\Retail Store Customer Analytics_ Footfall, Count, Age, Gender.mp4"
    path8 =r"E:\softweb\priyasoftweb\person_tracker_id_ft_age-gender_UBI-IOT\input\op_business-workers-posing-at-work-SEUYQKC.mp4"
    path9 =r"E:\softweb\priyasoftweb\person_tracker_id_ft_age-gender_UBI-IOT\input\op-family-buying-cheese-in-the-s_resized.mp4"
    path10 =r"E:\softweb\priyasoftweb\person_tracker_id_ft_age-gender_UBI-IOT\input\opp-family-buying-cheese-in-the-store-JSQKCZX.mp4"
    path11 =r"E:\softweb\priyasoftweb\person_tracker_id_ft_age-gender_UBI-IOT\input\op-family-buying-cheese-in-the-store-JSQKCZX.mp4"

    
    #MODEL_ROOT_PATH="./pretrain/"

    #face_path=r"E:\softweb\Pretrained_model\face_age_gender_model\yolov2_tiny-face.h5"
    age_path=r"E:\softweb\Pretrained_model\face_age_gender_model\agegender_age101_squeezenet.hdf5"
    gender_path=r"E:\softweb\Pretrained_model\face_age_gender_model\agegender_gender_squeezenet.hdf5"

    
    #Load Model
    #model_face = load_model(face_path)
    #model_age = load_model(MODEL_ROOT_PATH+'agegender_age_mobilenet_imdb.hdf5')
    model_age = load_model(age_path)
    model_gender = load_model(gender_path)

    #caffe face
        
    faceProto=r"E:\softweb\Pretrained_model\object_detectiom_model\deploy.prototxt.txt"
    faceModel=r"E:\softweb\Pretrained_model\object_detectiom_model\res10_300x300_ssd_iter_140000.caffemodel"
        
    faceNet=cv2.dnn.readNetFromCaffe(faceProto,faceModel)

    #Prepare WebCamer1
    cap = cv2.VideoCapture(path11)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    H=None
    W=None
    fourcc_codec = cv2.VideoWriter_fourcc(*'XVID')
    fps = 10.0
    capture_size = (int(cap.get(3)), int(cap.get(4)))
    
    out_ = cv2.VideoWriter("out_face_track(caffe)_age_gender(squeeznet)_v7.1.mp4", fourcc_codec, fps, capture_size)
    
    #Detection
    while True:
        #Face Detection
        ret, frame = cap.read() #BGR
        total_frames = total_frames + 1
        
        if not ret:
            break
        
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        resultImg,faceBoxes=highlightFace(faceNet,frame)
       
         #fps calcuation
    
        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)
    
        fps_text = "FPS: {:.2f}".format(fps)
    
        cv2.putText(resultImg, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    
        #Age and Gender Detection
        show_results(resultImg,frame,faceBoxes, model_age, model_gender, out_)
        
        
        k = cv2.waitKey(1)
        if k == 27:
            break
        
    cap.release()
    out_.release()
    cv2.destroyAllWindows()
    
if __name__=='__main__':
    main()