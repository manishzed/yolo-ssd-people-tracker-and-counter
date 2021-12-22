import os
import flask
from flask import Flask, request, render_template, send_file, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
# from imageai.Detection import ObjectDetection
import traceback
import pickle
import pandas as pd
import numpy as np
import imutils
#GENDER AND AGE DETECTION
import cv2
import math
import argparse
import base64
from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
from flask import request
from datetime import datetime

from flask_cors import CORS, cross_origin
import sys
import os
import datetime
import calendar
from PIL import Image
import pickle
import io
import json
app = Flask(__name__)

#gender and age detction------------

def highlightFace(net, frame, conf_threshold=0.5):
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
    
#gender and age detction

faceProto="model/opencv_face_detector.pbtxt"
faceModel="model/opencv_face_detector_uint8.pb"
ageProto="model/age_deploy.prototxt"
ageModel="model/age_net.caffemodel"
genderProto="model/gender_deploy.prototxt"
genderModel="model/gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)


padding=20      

# Take in base64 string and return PIL image
def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

# convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
def toRGB(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    
    
# POST - just get the image and metadata
@app.route('/encodeimage', methods=['GET','POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])

def encode_():
    dat = request.json
    image_64_encode =dat['img']
    #b = bytes(image_64_encode, encoding='ascii')
    im =stringToImage(image_64_encode)
    frame =toRGB(im)
    print(frame.shape)
    frame =imutils.resize(frame, width=600)
    print("!1111", frame.shape)
    #print("222222222222:", file)
    resultImg,faceBoxes=highlightFace(faceNet,frame)
    print("3333333333333:", faceBoxes)
    for faceBox in faceBoxes:
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
    
        #cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
       
    
        
        # In memory
        #image_content = cv2.imencode('.jpg', resultImg)[1].tostring()
        #encoded_image = base64.encodestring(image_content)
        #to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
        #to_send.save('test_image1.jpg')

    return ("Gender:{}, Age:{}".format(gender, age))
    
    
    


# POST - just get the image and metadata
@app.route('/imgupload', methods=['POST'])
def post():    
    file = request.files.get('image', '')
    print("fffff", file)
    #imagefile.save('test_image.jpg')
    #return "OK", 200
    # Read image
    frame = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    frame =imutils.resize(frame, width=600)
    print("!1111", frame.shape)
    print("222222222222:", file)
    resultImg,faceBoxes=highlightFace(faceNet,frame)
    print("3333333333333:", faceBoxes)
    for faceBox in faceBoxes:
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
    
        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
       
    
        
        # In memory
        image_content = cv2.imencode('.jpg', resultImg)[1].tostring()
        encoded_image = base64.encodestring(image_content)
        #to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
        #to_send.save('test_image1.jpg')
        da =encoded_image.decode('utf-8')
    return f'<img src="data:image/jpeg;base64,{da}">'



    
    
if __name__ == "__main__":
    app.run(debug=False)