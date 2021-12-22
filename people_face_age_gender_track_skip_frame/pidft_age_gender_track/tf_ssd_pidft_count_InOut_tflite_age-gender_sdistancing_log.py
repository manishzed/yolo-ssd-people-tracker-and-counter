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

from collections import defaultdict

from itertools import combinations
import math


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input", type=str,
    help="path to optional input video file")

args = vars(ap.parse_args())


thres = 0.5

#centroid
#tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)
#tracker = CentroidTracker(maxDisappeared=40, maxDistance=50)

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


#capture video from file
cap = cv2.VideoCapture(args["input"])
# get total number of frames
totalFrames_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print("totalFrames_count", totalFrames_count)
print('Position:', cap.get(cv2.CAP_PROP_POS_FRAMES))


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


#param
writer = None
W = None
H = None
conf=0.4
skip_frames=30

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0

x = []
empty=[]
empty1=[]

Log = True
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

#FPS
fps_start_time = datetime.datetime.now()
fps = 0
total_frames = 0
writer =None
frames_count =0

#frame time
object_id_list = []
dtime = dict()
dwell_time = dict()

gender_id=dict()
age_id =dict()
age_gender_id =dict()

fourcc_codec = cv2.VideoWriter_fourcc(*'XVID')
fps = 10.0
capture_size = (int(cap.get(3)), int(cap.get(4)))

out = cv2.VideoWriter("output_tf_ssd_pidft_tflite_age-gender_v7.avi", fourcc_codec, fps, capture_size)
while True:
    
    frame = cap.read()
    frame = frame[1] if args.get("input", False) else frame

    # if we are viewing a video and we did not grab a frame then we
    # have reached the end of the video
    if args["input"] is not None and frame is None:
        break

    # resize the frame to have a maximum width of 500 pixels (the
    # less data we have, the faster we can process it), then convert
    # the frame from BGR to RGB for dlib
    #frame = imutils.resize(frame, width=800)
        
    total_frames = total_frames + 1
    frame = imutils.resize(frame, width=800)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # if the frame dimensions are empty, set them
    if W is None or H is None:
        (H, W) = frame.shape[:2]


    # initialize the current status along with our list of bounding
    # box rectangles returned by either (1) our object detector or
    # (2) the correlation trackers
    status = "Waiting"
    rects = []

    # check to see if we should run a more computationally expensive
    # object detection method to aid our tracker
    if totalFrames % skip_frames == 0:
        print("skip_frames", skip_frames)
        print("totalFrames", totalFrames)
        # set the status and initialize our new set of object trackers
        status = "Detecting"
        trackers = []

        classIds, confs, bbox = net.detect(frame, confThreshold=thres)
        
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
                print(nms_confs)
                #print(nms_clsIds)
                #print(classNames[i])
                #print(nms_box)
                if nms_confs>0.5:
                    
                    if classNames[nms_classIds-1]!="person":
                        continue
                
                    #x, y, w, h = nms_bbox
                    (startX, startY, endX, endY) = nms_bbox.astype("int")
                    print(startX, startY, endX, endY)

                    # construct a dlib rectangle object from the bounding
                    # box coordinates and then start the dlib correlation
                    # tracker
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)
    
                    # add the tracker to our list of trackers so we can
                    # utilize it during skip frames
                    trackers.append(tracker)
    # otherwise, we should utilize our object *trackers* rather than
    # object *detectors* to obtain a higher frame processing throughput
    else:
        # loop over the trackers
        for tracker in trackers:
            # set the status of our system to be 'tracking' rather
            # than 'waiting' or 'detecting'
            status = "Tracking"

            # update the tracker and grab the updated position
            tracker.update(rgb)
            pos = tracker.get_position()

            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # add the bounding box coordinates to the rectangles list
            rects.append((startX, startY, endX, endY))

    # draw a horizontal line in the center of the frame -- once an
    # object crosses this line we will determine whether they were
    # moving 'up' , "left" or 'down' , "right"
    cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
   
    # use the centroid tracker to associate the (1) old object
    # centroids with (2) the newly computed object centroids              
    print("rects",rects)
    boundingboxes = np.array(rects)
    boundingboxes = boundingboxes.astype(int)
    rects = non_max_suppression_fast(boundingboxes, 0.3)
    #tracker
    objects = ct.update(rects)
    #print(objects)
    objectId_ls =[]

    print("objects", objects)
    print("rects:", rects)
    
    centroid_dict =dict()
    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        print("objid", objectID)
        print("centroid", centroid)
        
        x1, y1, x2, y2 = centroid
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        
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
            
        # check to see if a trackable object exists for the current
        # object ID
        to = trackableObjects.get(objectID, None)

        # if there is no existing trackable object, create one
        if to is None:
            to = TrackableObject(objectID, centroid)

        # otherwise, there is a trackable object so we can utilize it
        # to determine direction
        else:
            # the difference between the y-coordinate of the *current*
            # centroid and the mean of *previous* centroids will tell
            # us in which direction the object is moving (negative for
            # 'up' and positive for 'down')
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            # check to see if the object has been counted or not
            if not to.counted:
                # if the direction is negative (indicating the object
                # is moving up) AND the centroid is above the center
                # line, count the object
                if direction < 0 and centroid[1] < H // 2:
                    totalUp += 1
                    empty.append(totalUp)
                    to.counted = True

                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                elif direction > 0 and centroid[1] > H // 2:
                    totalDown += 1
                    empty1.append(totalDown)
                    to.counted = True
                
                    
                x = []
                # compute the sum of total people inside
                x.append(len(empty1)-len(empty))
                #print("Total people inside:", x)

        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        #frame time ----end--------
    
        #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        #text = "ID: {}".format(objectID)
        #text = "id:{},ft:{}sec".format(objectID, int(dwell_time[objectID]))
        #cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        
        #social distancing
        cX = int((x1 + x2) / 2.0)
        cY = int((y1 + y2) / 2.0)
        centroid_dict[objectID] = (cX, cY, x1, y1, x2, y2)
    
        red_zone_list = []
        for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2):
            dx, dy = p1[0] - p2[0], p1[1] - p2[1]
            distance = math.sqrt(dx * dx + dy * dy)
            if distance < 75.0:
                if id1 not in red_zone_list:
                    red_zone_list.append(id1)
                if id2 not in red_zone_list:
                    red_zone_list.append(id2)
    
        for id, box in centroid_dict.items():
            if id in red_zone_list:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = "id:{}ft:{}".format(objectID, int(dwell_time[objectID]))
            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            
        
        #222222----start------age-gender-----
        #tflite age-egnder--part2-----start----------
        
        saved_image = frame   
        #x, y, w, h=x1, y1, x2, y2
        input_im = saved_image[y1:y1+y2, x1:x1+x2]
        
        if input_im is None:
            print("input not getting")
        else:
            input_im = cv2.resize(input_im, (128,128))
            #input_im = cv2.resize(input_im, (128,128), fx=0,fy=0, interpolation = cv2.INTER_CUBIC)

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


            #age
            output_data_age =output_data_age.reshape(-1)
            
            age=0
            for i in range(101):
                age=age+output_data_age[i]*i
                
            label=str(int(age))
            
            age_gen_text = "{}|{}".format(label, prezic_gender)
            cv2.putText(frame, age_gen_text, (x1, y1-20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)


    
    # construct a tuple of information we will be displaying on the
    # frame
    #count perosn in and out
    info = [
        ("Right", totalUp),
        ("Left", totalDown),
    ]
    
    #count inside
    info2 = [
        ("Total people inside", x),
        ]
    
    print(info)
    # loop over the info tuples and draw them on our frame
    for (i, (k, v)) in enumerate(info):
        text_ = "{}: {}".format(k, v)
        cv2.putText(frame, text_, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 255), 2)
    
    
    for (i, (k, v)) in enumerate(info2):
        
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (265, H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


    #save log data of person in and out
    # Initiate a simple log to save data at end of the day
    if Log:
        datetimee = [datetime.datetime.now()]
        d = [datetimee, empty1, empty, x]
        export_data = zip_longest(*d, fillvalue = '')

        with open('Log.csv', 'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(("End Time", "In", "Out", "Total Inside"))
            wr.writerows(export_data)
    #fps
    
    fps_end_time = datetime.datetime.now()
    time_diff = fps_end_time - fps_start_time
    if time_diff.seconds == 0:
        fps = 0.0
    else:
        fps = (total_frames / time_diff.seconds)

    fps_text = "FPS: {:.2f}".format(fps)

    cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        

    #count current perosn count and total person count
    lpc_count = len(objects)
    opc_count = len(object_id_list)

    lpc_txt = "LPC: {}".format(lpc_count)
    opc_txt = "OPC: {}".format(opc_count)

    cv2.putText(frame, lpc_txt, (5, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    cv2.putText(frame, opc_txt, (5, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    #write
    #out.write(frame)
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # increment the total number of frames processed thus far and
    # then update the FPS counter
    totalFrames += 1


# close any open windows
cap.release()
#out.close()
cv2.destroyAllWindows()