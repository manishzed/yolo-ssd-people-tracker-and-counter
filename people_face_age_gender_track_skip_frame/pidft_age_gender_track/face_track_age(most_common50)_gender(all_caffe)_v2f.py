import cv2
import math
import argparse
import datetime
from centroidtracker import CentroidTracker
from collections import defaultdict

tracker = CentroidTracker(maxDisappeared=40, maxDistance=50)

#face age and gender
faceProto=r"E:\softweb\Pretrained_model\object_detectiom_model\deploy.prototxt.txt"
faceModel=r"E:\softweb\Pretrained_model\object_detectiom_model\res10_300x300_ssd_iter_140000.caffemodel"

ageProto=r"E:\softweb\Pretrained_model\face_age_gender_model\age_deploy.prototxt"
ageModel=r"E:\softweb\Pretrained_model\face_age_gender_model\age_net.caffemodel"
genderProto=r"E:\softweb\Pretrained_model\face_age_gender_model\gender_deploy.prototxt"
genderModel=r"E:\softweb\Pretrained_model\face_age_gender_model\gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['M','F']

faceNet=cv2.dnn.readNetFromCaffe(faceProto,faceModel)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)



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

cap=cv2.VideoCapture(path11)

padding=20
conf_threshold=0.7

fps_start_time = datetime.datetime.now()
fps = 0
total_frames = 0

#update var
frames_count =0

#frame time
object_id_list = []
object_id_list_gender=[]
object_id_list_age=[]
object_id_lis_age_gendert=[]

dtime = dict()
dwell_time = dict()

age_id =dict()
gender_id=dict()
age_gender_id=dict()
gender_id_dict =dict()

#test var
age_id_dict_list = defaultdict(list)
dict_age_=defaultdict(list)

#save video
fourcc_codec = cv2.VideoWriter_fourcc(*'XVID')
fps = 10.0
capture_size = (int(cap.get(3)), int(cap.get(4)))

out_ = cv2.VideoWriter("out_face_track_gender_age(caffe)_v2.1.mp4", fourcc_codec, fps, capture_size)

print("!!!!!!!!!!!!!!!!!!!!!!!1")
while True:
    ret,frame=cap.read()
    print("222222222222222222222")
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
            


        cv2.rectangle(frameOpencvDnn, (x11, y11), (x22, y22), (255, 255, 0), 2)
        #text = "ID: {}".format(objectId)
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
        prezic_gender = genderList[genderPreds[0].argmax()]
        #print(f'Gender: {prezic_gender}')

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        prezic_age = ageList[agePreds[0].argmax()]
        #print(f'Age: {prezic_age[1:-1]} years')
        
        
        #custom update age
        print("{}{}{}{}".format(objectId, prezic_gender, prezic_age, frames_count))
        #gender_id_dict[objectId]=prezic_gender
        #age_id[objectId] =[prezic_age]
        
        
        age_id_dict_list[objectId].append(prezic_age)
        #print("age_id_dict_list", age_id_dict_list)
        
        
        
        from collections import defaultdict
        from collections import Counter
        
        dict_age_most_comm=defaultdict(list)
        dict_age_all=defaultdict(list)
        
        for k, v in age_id_dict_list.items():
            s =Counter(v)
            dict_age_most_comm[k].append(s.most_common())
            dict_age_all[k].append(s)
        
        print("dict_age_most_comm", dict_age_most_comm)
        print("dict_age_all",dict_age_all)
        
        
        dict_age_50 = defaultdict(list)
        
        for u in range(len(dict_age_all)):
            #print(u)
            
            for k4, v4 in dict_age_all[u][0].items():
                #print(k4)
                #if v4>=15 and v4 <=50:
                if v4>=50:
                    #print(k4)
                    dict_age_50[u].append((k4,v4))

                    
        print("dict_age_50",dict_age_50)
        
        dict_age_50_max =defaultdict(list)
        dic_max_f =defaultdict(list)
        for d in range(len(dict_age_50)):
            #print(max(dict_age_50[d]))
            sx= dict(dict_age_50[d])
            print(sx)
            if sx=={}:
                continue
            sxxx =max(sx.items(), key=lambda k: k[1])
            
            dic_max_f[d]=sxxx
            
        print("dic_max_f", dic_max_f)
            
            
        
    
        #print("dict_age_most_comm", dict_age_most_comm)
        #print("dict_age_all",dict_age_all)
        #print('dict_age_50', dict_age_50)
        #print("dic_max_f", dic_max_f)

        #cv2.putText(frameOpencvDnn, f'{prezic_gender}, {prezic_age}', (x11, y11-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 1)
        
        if dic_max_f[objectId]==[]:
            continue
        print("dic_max_f[objectId][0]", dic_max_f[objectId])

        cv2.putText(frameOpencvDnn, f'{prezic_gender}, {dic_max_f[objectId][0]}', (x11, y11-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 1)

        #cv2.imshow("Detecting age and gender", resultImg)
    
        
    #fps
    
    fps_end_time = datetime.datetime.now()
    time_diff = fps_end_time - fps_start_time
    if time_diff.seconds == 0:
        fps = 0.0
    else:
        fps = (total_frames / time_diff.seconds)

    fps_text = "FPS: {:.2f}".format(fps)

    cv2.putText(frameOpencvDnn, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

    #write
    out_.write(frameOpencvDnn)
    
    cv2.imshow("Application", frameOpencvDnn)
    
    frames_count =frames_count+1
    print("frames_count", frames_count)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

print("t_frames_count", frames_count)
#print("age_id_dict_list", age_id_dict_list)


cap.release()
out_.release()
cv2.destroyAllWindows()
