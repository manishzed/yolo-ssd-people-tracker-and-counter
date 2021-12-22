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
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (255,255,0), int(round(frameHeight/150)), 5)
    return frameOpencvDnn,faceBoxes

#display result
def show_results(resultImg,frame, faceBoxes, model_age, model_gender,out_):
    
    padding=20
    
    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        face_image = frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]
                
            
        #lines_age=open('words/agegender_age_words.txt').readlines()
        #lines_gender=open('words/agegender_gender_words.txt').readlines()
        
        lines_age =['(0, 2)','(4, 6)','(8, 13)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
        lines_gender = ['F', 'M']
        
        if(model_age!=None):
            shape = model_age.layers[0].get_output_at(0).get_shape().as_list()
            img_keras = cv2.resize(face_image, (shape[1],shape[2]))
            #img_keras = img_keras[::-1, :, ::-1].copy()    #BGR to RGB
            img_keras = np.expand_dims(img_keras, axis=0)
            img_keras = img_keras / 255.0
            
            pred_age_keras = model_age.predict(img_keras)[0]
            prob_age_keras = np.max(pred_age_keras)
            cls_age_keras = pred_age_keras.argmax()
    
            age=0
            for i in range(101):
                age=age+pred_age_keras[i]*i
            label=str(int(age))
            
            #label="%.2f" % prob_age_keras + " " + lines_age[cls_age_keras]
            print(cls_age_keras)
    
            cv2.putText(resultImg, "Age:"+label, (faceBox[0],faceBox[1]-30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0,0,250));
    
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
            cv2.putText(resultImg, "Gender:"+ lines_gender[cls_gender_keras], (faceBox[0],faceBox[1]-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0,0,250));
        
    out_.write(resultImg)
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

    #Prepare WebCamera
    cap = cv2.VideoCapture(path9)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    H=None
    W=None
    fourcc_codec = cv2.VideoWriter_fourcc(*'XVID')
    fps = 10.0
    capture_size = (int(cap.get(3)), int(cap.get(4)))
    
    out_ = cv2.VideoWriter("out_face(caffe)_age_gender(squeeznet)_v1.4.mp4", fourcc_codec, fps, capture_size)
    
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