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

def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))


#age-gender---startt--111111------
#tflite age---gender---start-------------------

string_pred_gen = ['F', 'M']

# Load TFLite model and allocate tensors. Load Face Cascade
#face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

age_path =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\age-gender\age-gender-.h5-keras\preatrain_50epoch-age101\agegender_age101_mobilenet_imdb.tflite"
interpreter_age = tf.lite.Interpreter(model_path=age_path)
interpreter_age.allocate_tensors()

gender_path =age_path =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\age-gender\age-gender-.h5-keras\preatrain_50epoch-age101\agegender_gender_mobilenet_imdb.tflite"

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

#path1 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\test_video.mp4"

path2 = r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\VPD officers target violent shoplifters in month-long initiative - Vancouver Sun.mp4"
path3 = r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\VPD_part1.mp4"
path4= r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\VPD_part2.mp4"
path33= r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\store_camv1.mp4"
path5=r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\motor_bike.mp4"
path6 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\example_01.mp4"
path7 =r"E:\softweb\priyasoftweb\FaceDetection-covid19\face_mask_detection_keras-social_distance\input\CDC recommends wearing face masks in public.mp4"
path8= r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\store_camv1.mp4"
path9=r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\store_amazon_go.mp4"
cap = cv2.VideoCapture(path8)

# get total number of frames
totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT) 

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

#frame time
object_id_list = []
dtime = dict()
dwell_time = dict()

gender_id=dict()

fourcc_codec = cv2.VideoWriter_fourcc(*'XVID')
fps = 10.0
capture_size = (int(cap.get(3)), int(cap.get(4)))

out = cv2.VideoWriter("output_tf_ssd_pidft_tflite_age-gender_v7.avi", fourcc_codec, fps, capture_size)
while True:
    
    ret, img = cap.read()
    
    if ret:
        #img =imutils.resize(img, width=800)
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
                #rects.append((x, y, w, h))
                rects.append(nms_bbox)
                
        
        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.3)
        #tracker
        objects = tracker.update(rects)
        #print(objects)
        objectId_ls =[]
        
        for (objectID, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
    
            objectId_ls.append(objectID)
            
             #frame time ------start----
            if objectID not in object_id_list:
                object_id_list.append(objectID)
                dtime[objectID] = datetime.datetime.now()
                dwell_time[objectID] = 0
            else:
                curr_time = datetime.datetime.now()
                old_time = dtime[objectID]
                time_diff = curr_time - old_time
                dtime[objectID] = datetime.datetime.now()
                sec = time_diff.total_seconds()
                dwell_time[objectID] += sec
    
            cv2.rectangle(img, (x1, y1), (x1+x2, y1+y2), (0, 0, 255), 2)
            text = "id:{},ft:{}sec".format(objectID, int(dwell_time[objectID]))
            cv2.putText(img, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            
            #222222----start------age-gender-----
            #tflite age-egnder--part2-----start----------
            
            saved_image = img   
            #x, y, w, h=x1, y1, x2, y2
            input_im = saved_image[y1:y1+y2, x1:x1+x2]
            
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
                #index_pred_age = int(np.argmax(output_data_age))
                index_pred_gender = int(np.argmax(output_data_gender))
                #prezic_age = string_pred_age[index_pred_age]
                prezic_gender = string_pred_gen[index_pred_gender]
    
                print("id{}gen{}".format(objectID, prezic_gender))
                gender_id[objectID]=prezic_gender
                #age
                output_data_age =output_data_age.reshape(-1)
                
                age=0
                for i in range(101):
                    age=age+output_data_age[i]*i
                    
                label=str(int(age))
                
                age_gen_text = "{}|{}".format(label, prezic_gender)
                cv2.putText(img, age_gen_text, (x1, y1-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    
        
                
        #FPS
        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)
    
        fps_text = "FPS: {:.2f}".format(fps)
    
        cv2.putText(img, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
        
        out.write(img)
        #show images
        cv2.imshow("img", img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        print("get image failed")
        break

cap.release()
out.release()
cv2.destroyAllWindows()
