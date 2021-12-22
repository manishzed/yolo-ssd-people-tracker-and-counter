import cv2
import math
import argparse
import datetime
import imutils
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np



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
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes


faceProto=r"E:\softweb\Pretrained_model\object_detectiom_model\deploy.prototxt.txt"
faceModel=r"E:\softweb\Pretrained_model\object_detectiom_model\res10_300x300_ssd_iter_140000.caffemodel"

genderProto=r"E:\softweb\Pretrained_model\face_age_gender_model\gender_deploy.prototxt"
genderModel=r"E:\softweb\Pretrained_model\face_age_gender_model\gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
genderList=['Male','Female']

faceNet=cv2.dnn.readNetFromCaffe(faceProto,faceModel)
genderNet=cv2.dnn.readNet(genderModel,genderProto)


#age gender model
#tflite age---gender---start-------------------

string_pred_age =['(0, 2)','(4, 6)','(8, 13)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']

age_path =r"E:\softweb\priyasoftweb\person_tracker_id_ft_age-gender_UBI-IOT\preatrain_50epoch-age101\agegender_age_mobilenet_imdb.tflite"
interpreter_age = tf.lite.Interpreter(model_path=age_path)
interpreter_age.allocate_tensors()


# # Get input and output tensors
input_details_age = interpreter_age.get_input_details()
output_details_age = interpreter_age.get_output_details()
input_shape_age = input_details_age[0]['shape']


input_im = None
font = cv2.FONT_HERSHEY_PLAIN
#tflite age-gender --------end-----------


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

fps_start_time = datetime.datetime.now()
fps = 0
total_frames = 0

W =None
H =None

fourcc_codec = cv2.VideoWriter_fourcc(*'XVID')
fps = 10.0
capture_size = (int(cap.get(3)), int(cap.get(4)))

out_ = cv2.VideoWriter("out_face_gender(caffe)_age(mobilenet)_v14.mp4", fourcc_codec, fps, capture_size)

while True:
    ret,frame=cap.read()
    #frame =imutils.resize(frame, width=800)
    total_frames = total_frames + 1
    
    if not ret:
        break
    
    resultImg,faceBoxes=highlightFace(faceNet,frame)
    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        input_im = frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]
            
        #gender
        blob=cv2.dnn.blobFromImage(input_im, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        prezic_gender = genderList[genderPreds[0].argmax()]
        print(f'Gender: {prezic_gender}')
        
        #age
        input_im = cv2.resize(input_im, (128,128))
        input_im = input_im.astype('float')
        input_im = input_im / 255
        input_im = img_to_array(input_im)
        input_im = np.expand_dims(input_im, axis = 0)

        # Predict
        input_data = np.array(input_im, dtype=np.float32)
        interpreter_age.set_tensor(input_details_age[0]['index'], input_data)
        interpreter_age.invoke()

        output_data_age = interpreter_age.get_tensor(output_details_age[0]['index'])
        index_pred_age = int(np.argmax(output_data_age))
        prezic_age = string_pred_age[index_pred_age]

            
        #age_gen_text = "{}|{}".format(label, prezic_gender)
        #cv2.putText(frame, age_gen_text, (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)


        #age_gen_text = "{}|{}".format(prezic_age, prezic_gender)
        #cv2.putText(img, age_gen_text, (x1, y1-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        #cv2.rectangle(img, (xx, yy), (xx+ww, yy+hh), (0, 0, 255), 2)
        cv2.putText(resultImg, f'{prezic_gender}, {prezic_age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

    
    #fps
    
    fps_end_time = datetime.datetime.now()
    time_diff = fps_end_time - fps_start_time
    if time_diff.seconds == 0:
        fps = 0.0
    else:
        fps = (total_frames / time_diff.seconds)

    fps_text = "FPS: {:.2f}".format(fps)

    cv2.putText(resultImg, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

    out_.write(resultImg)
    cv2.imshow("Application", resultImg)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
out_.release()
cv2.destroyAllWindows()
