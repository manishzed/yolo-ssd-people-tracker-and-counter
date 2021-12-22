
import numpy as np
import cv2
from time import time
import imutils

# Load TFLite model and allocate tensors. Load Face Cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


path1 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\face-demographics-walking.mp4"
path2 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\face-demographics-walking-and-pause.mp4"
path3 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\happy-female-customers-choosing-makeup-products-SUNT7GD_2.mp4"

webcam = cv2.VideoCapture(path3)
# =============================================================================
# # Setare rezolutie camera Raspberry (320x240, 640x480, 1280x720, 1920x1080-nerecomandat)
# webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# 
# =============================================================================

while True:
    font = cv2.FONT_HERSHEY_PLAIN
    time_start = time()
    _, frame = webcam.read()
    
    frame =imutils.resize(frame, width=800)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor =2, minNeighbors=10)
    print(faces)
    for x,y,w,h in faces:
        
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,255), 1)

    fps = "FPS: " + str(round(1.0 / (time() - time_start), 2))      
    cv2.putText(frame, fps, (20,20), font, 1, (250,250,250), 1, cv2.LINE_AA)     
    
    cv2.imshow("Detecting faces...", frame)
    
    key = cv2.waitKey(1)

    if key == ord('q') or key == ord('Q'):
        webcam.release()
        cv2.destroyAllWindows()
        break
    
    
#upgrade

import numpy as np
import cv2
from time import time
import imutils

# Load TFLite model and allocate tensors. Load Face Cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


path1 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\face-demographics-walking.mp4"
path2 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\face-demographics-walking-and-pause.mp4"
path3 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\happy_customer.mp4"

webcam = cv2.VideoCapture(path3)

    
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
    
while True:
    font = cv2.FONT_HERSHEY_PLAIN
    time_start = time()
    _, frame = webcam.read()
    
    frame =imutils.resize(frame, width=800)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor = 1.5, minNeighbors=8)
    print(faces)
    
    
    rects =[]
    for face_cor in faces:
        
        (startX, startY, endX, endY) = face_cor.astype("int")
        rects.append((startX, startY, endX, endY))
        
    boundingboxes = np.array(rects)
    boundingboxes = boundingboxes.astype(int)
    rects = non_max_suppression_fast(boundingboxes, 0.3)
    
    for coordinate in rects:
        (x, y, w, h) = coordinate
        colors = np.random.randint(1, 255, 3)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (int(colors[0]), int(colors[1]), int(colors[2])), thickness=2)

    fps = "FPS: " + str(round(1.0 / (time() - time_start), 2))      
    cv2.putText(frame, fps, (20,20), font, 1, (250,250,250), 1, cv2.LINE_AA)     
    
    cv2.imshow("Detecting faces...", frame)
    
    key = cv2.waitKey(1)

    if key == ord('q') or key == ord('Q'):
        webcam.release()
        cv2.destroyAllWindows()
        break
    
    
    
    
    
    

import numpy as np
import cv2
from time import time
import imutils

# Load TFLite model and allocate tensors. Load Face Cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


path1 ="input\face-demographics-walking-and-pause.mp4"
#path2 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\face-demographics-walking-and-pause.mp4"
#path3 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\happy-female-customers-choosing-makeup-products-SUNT7GD_2.mp4"

webcam = cv2.VideoCapture(path1)
# =============================================================================
# # Setare rezolutie camera Raspberry (320x240, 640x480, 1280x720, 1920x1080-nerecomandat)
# webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# 
# =============================================================================

while True:
    font = cv2.FONT_HERSHEY_PLAIN
    time_start = time()
    _, frame = webcam.read()
    
    frame =imutils.resize(frame, width=800)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor =2, minNeighbors=10)
    print(faces)
    for x,y,w,h in faces:
        
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,255), 1)

    fps = "FPS: " + str(round(1.0 / (time() - time_start), 2))      
    cv2.putText(frame, fps, (20,20), font, 1, (250,250,250), 1, cv2.LINE_AA)     
    
    cv2.imshow("Detecting faces...", frame)
    
    key = cv2.waitKey(1)

    if key == ord('q') or key == ord('Q'):
        webcam.release()
        cv2.destroyAllWindows()
        break
    
