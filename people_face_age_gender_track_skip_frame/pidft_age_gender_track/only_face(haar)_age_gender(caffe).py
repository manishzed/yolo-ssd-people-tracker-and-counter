
import cv2
import math
import argparse
import datetime

ageProto=r"E:\softweb\Pretrained_model\face_age_gender_model\age_deploy.prototxt"
ageModel=r"E:\softweb\Pretrained_model\face_age_gender_model\age_net.caffemodel"
genderProto=r"E:\softweb\Pretrained_model\face_age_gender_model\gender_deploy.prototxt"
genderModel=r"E:\softweb\Pretrained_model\face_age_gender_model\gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['M','F']

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

path12 =r"E:\softweb\priyasoftweb\person_tracker_id_ft_age-gender_UBI-IOT\input\WIN_20210720_17_54_55_Pro.mp4"
path13 =r"E:\softweb\priyasoftweb\person_tracker_id_ft_age-gender_UBI-IOT\input\WIN_20210720_17_56_29_Pro.mp4"

path14 =r"E:\softweb\priyasoftweb\person_tracker_id_ft_age-gender_UBI-IOT\input\head-pose-face-detection-female.mp4"

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
input_im=None


cap=cv2.VideoCapture(path14)
padding=20

fps_start_time = datetime.datetime.now()
fps = 0
total_frames = 0

#save video
fourcc_codec = cv2.VideoWriter_fourcc(*'XVID')
fps = 10.0
capture_size = (int(cap.get(3)), int(cap.get(4)))

out_ = cv2.VideoWriter("out_demog_(haar)_head-pose-face-detection-female.mp4", fourcc_codec, fps, capture_size)

while True :
    ret,frame=cap.read()
    total_frames = total_frames + 1

    if not ret:
        break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor = 1.5, minNeighbors=8)
    for xx,yy,ww,hh in faces:
        saved_image = frame          
        input_im = saved_image[yy:yy+hh, xx:xx+ww]
        
        if input_im is None:
            print("No face detected")

        else:
            blob=cv2.dnn.blobFromImage(input_im, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds=genderNet.forward()
            gender=genderList[genderPreds[0].argmax()]
            print(f'Gender: {gender}')
    
            ageNet.setInput(blob)
            agePreds=ageNet.forward()
            age=ageList[agePreds[0].argmax()]
            print(f'Age: {age[1:-1]} years')
    
            cv2.putText(frame, f'{gender}, {age}', (xx, yy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
            #cv2.imshow("Detecting age and gender", resultImg)
            cv2.rectangle(frame, (xx, yy), (xx+ww, yy+hh), (0, 255, 0), 2)
            
            
    #fps
    
    fps_end_time = datetime.datetime.now()
    time_diff = fps_end_time - fps_start_time
    if time_diff.seconds == 0:
        fps = 0.0
    else:
        fps = (total_frames / time_diff.seconds)

    fps_text = "FPS: {:.2f}".format(fps)

    cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    
    out_.write(frame)
    cv2.imshow("Application", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
out_.release()
cv2.destroyAllWindows()