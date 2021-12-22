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
from tensorflow.keras.preprocessing.image import img_to_array
import time, csv
import numpy as np
import argparse, imutils
import time, dlib, cv2, datetime
from itertools import zip_longest
import tensorflow as tf

thres = 0.5
#centroid
#tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)
tracker = CentroidTracker(maxDisappeared=40, maxDistance=50)

#age-gender---startt--111111------
#tflite age---gender---start-------------------
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

string_pred_age =['(0, 2)','(4, 6)','(8, 13)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']

string_pred_gen = ['F', 'M']

# Load TFLite model and allocate tensors. Load Face Cascade
#face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

age_path =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\age-gender\age-gender-.h5-keras\preatrain_50epoch-age101\agegender_age_mobilenet_imdb.tflite"
interpreter_age = tf.lite.Interpreter(model_path=age_path)
interpreter_age.allocate_tensors()

gender_path =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\age-gender\age-gender-.h5-keras\preatrain_50epoch-age101\agegender_gender_mobilenet_imdb.tflite"

interpreter_gender = tf.lite.Interpreter(model_path=gender_path)
interpreter_gender.allocate_tensors()

# # Get input and output tensors
input_details_age = interpreter_age.get_input_details()
output_details_age = interpreter_age.get_output_details()
input_shape_age = input_details_age[0]['shape']

input_details_gender = interpreter_gender.get_input_details()
output_details_gender = interpreter_gender.get_output_details()
input_shape_gender = input_details_gender[0]['shape']

input_im = None
font = cv2.FONT_HERSHEY_PLAIN
#tflite age-gender --------end-----------

#1111---end------age-gender-------



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


path10 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\multi_face2.jpg"

path33= r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\store_camv1.mp4"
path5=r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\motor_bike.mp4"
path6 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\example_01.mp4"
path7 =r"E:\softweb\priyasoftweb\FaceDetection-covid19\face_mask_detection_keras-social_distance\input\CDC recommends wearing face masks in public.mp4"
path9=r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\store_amazon_go.mp4"

ubi_path1 =r"C:\Users\hp\Desktop\priyasoftweb\UBI-IOT\ubi\8_2021-05-26_14-33-26.mp4"
ubi_path2=r"C:\Users\hp\Desktop\priyasoftweb\UBI-IOT\ubi\8_2021-05-26_14-31-52.mp4"

ubi_path3 = r"C:\Users\hp\Desktop\priyasoftweb\UBI-IOT\ubi\8_2021-05-26_14-31-01.mp4"

cap = cv2.VideoCapture(ubi_path2)
 
fps_start_time = datetime.datetime.now()
fps = 0
total_frames = 0
writer =None

frames_count=0
while True:
    ret, img =cap.read()
    
    if not ret:
        break
    
    total_frames =total_frames+1
    img =imutils.resize(img, width=800)
    
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))
    #print(type(confs[0]))
    #print(confs)
    nms_indices =cv2.dnn.NMSBoxes(bbox, confs, thres,nms_threshold =0.2)
    #print(nms_indices)
    rects =[]
    object_id_list = []
    
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
    for (objectID, bbox) in objects.items():
        x1, y1, x2, y2 = bbox
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
    
        objectId_ls.append(objectID)
    
        cv2.rectangle(img, (x1, y1), (x1+x2, y1+y2), (0, 0, 255), 2)
        text = "id:{}".format(objectID)
        cv2.putText(img, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        
        #222222----start------age-gender-----
        #tflite age-egnder--part2-----start----------
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor = 1.2, minNeighbors=5)
        for xx,yy,ww,hh in faces:
    
            saved_image = img   
            #x, y, w, h=x1, y1, x2, y2
            #input_im = saved_image[y1:y1+y2, x1:x1+x2]
            input_im = saved_image[yy:yy+hh, xx:xx+ww]
    
            
            if input_im is None:
                print("input not getting")
            else:
                input_im = cv2.resize(input_im, (128,128))
                input_im = input_im.astype('float')
                input_im = input_im / 255
                input_im = img_to_array(input_im)
                input_im = np.expand_dims(input_im, axis = 0)
        
                 # Predict
                input_data = np.array(input_im, dtype=np.float32)
                interpreter_age.set_tensor(input_details_age[0]['index'], input_data)
                interpreter_age.invoke()
                interpreter_gender.set_tensor(input_details_gender[0]['index'], input_data)
                interpreter_gender.invoke()
    
                output_data_age = interpreter_age.get_tensor(output_details_age[0]['index'])
                output_data_gender = interpreter_gender.get_tensor(output_details_gender[0]['index'])
                index_pred_age = int(np.argmax(output_data_age))
                index_pred_gender = int(np.argmax(output_data_gender))
                prezic_age = string_pred_age[index_pred_age]
                prezic_gender = string_pred_gen[index_pred_gender]
        
                
                age_gen_text = "{}|{}".format(prezic_age, prezic_gender)
                #for face
                #cv2.putText(img, age_gen_text, (xx, yy), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                cv2.putText(img, age_gen_text, (x1, y1-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                cv2.rectangle(img, (xx, yy), (xx+ww, yy+hh), (0, 255, 0), 2)
   
    #FPS
    fps_end_time = datetime.datetime.now()
    time_diff = fps_end_time - fps_start_time
    if time_diff.seconds == 0:
        fps = 0.0
    else:
        fps = (total_frames / time_diff.seconds)

    fps_text = "FPS: {:.2f}".format(fps)

    cv2.putText(img, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
    

    #show images
    cv2.imshow("img", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
    frames_count =frames_count +1
    print("frames_count",frames_count)


print("total_frames_count",frames_count)

cap.release()

cv2.destroyAllWindows()
