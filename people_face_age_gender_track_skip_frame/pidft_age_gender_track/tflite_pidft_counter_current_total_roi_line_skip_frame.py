from centroidtracker import CentroidTracker
from trackableobject import TrackableObject
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import datetime
import tensorflow
import cv2
import datetime
import imutils
import numpy as np
from centroidtracker import CentroidTracker
import tensorflow as tf # version 1.14
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from time import time



#age-gender---startt--111111------
#tflite age---gender---start-------------------

string_pred_gen = ['F', 'M']

# Load TFLite model and allocate tensors. Load Face Cascade
#face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

age_path =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\age-gender\age-gender-.h5-keras\preatrain_50epoch-age101\agegender_age101_mobilenet_imdb.tflite"
interpreter_age = tensorflow.lite.Interpreter(age_path)
interpreter_age.allocate_tensors()

gender_path  =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\age-gender\age-gender-.h5-keras\preatrain_50epoch-age101\agegender_gender_mobilenet_imdb.tflite"

interpreter_gender = tensorflow.lite.Interpreter(gender_path)
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


threshold=0.55
min_conf_threshold = float(threshold)

# Path to video file
VIDEO_PATH = r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\VPD_part2.mp4"
path33= r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\store_camv1.mp4"
path5=r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\motor_bike.mp4"
path6 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\example_01.mp4"
path7 =r"E:\softweb\priyasoftweb\FaceDetection-covid19\face_mask_detection_keras-social_distance\input\CDC recommends wearing face masks in public.mp4"

ubi_path1 =r"C:\Users\hp\Desktop\priyasoftweb\UBI-IOT\data\ubi\8_2021-05-26_14-33-26.mp4"
ubi_path2=r"C:\Users\hp\Desktop\priyasoftweb\UBI-IOT\data\ubi\8_2021-05-26_14-31-52.mp4"
ubi_path3=r"C:\Users\hp\Desktop\priyasoftweb\UBI-IOT\data\ubi\8_2021-05-26_14-30-47.mp4"
ubi_path4=r"C:\Users\hp\Desktop\priyasoftweb\UBI-IOT\data\ubi\8_2021-05-26_14-30-10.mp4"
ubi_path5=r"C:\Users\hp\Desktop\priyasoftweb\UBI-IOT\data\ubi\8_2021-05-26_14-28-33.mp4"

#path video
path1 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\face-demographics-walking.mp4"
path2 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\face-demographics-walking-and-pause.mp4"
path3 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\happy-female-customers-choosing-makeup-products-SUNT7GD_2.mp4"
path4 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\example_01.mp4"

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = r"C:\Users\hp\Desktop\priyasoftweb\UBI-IOT\model\detect.tflite"

# Path to label map file
PATH_TO_LABELS = r"C:\Users\hp\Desktop\priyasoftweb\UBI-IOT\model\coco.names"

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])


interpreter = tf.lite.Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Open video file
cap = cv2.VideoCapture(path4)
imW = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#capture video from file

W =None 
H = None
skip_frames=30
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# initialize the total number of frames processed thus far, along
# with the total number of objects that have moved either up or down
totalFrames = 0
totalDown = 0
totalUp = 0

#frame time
object_id_list = []
dtime = dict()
dwell_time = dict()

#fps
fps_start_time = datetime.datetime.now()
fps = 0
total_frames = 0


while True:

    ret,frame = cap.read()
    
    # resize the frame to have a maximum width of 500 pixels (the
    # less data we have, the faster we can process it), then convert
    # the frame from BGR to RGB for dlib
    #frame = imutils.resize(frame, width=800)
        
    total_frames = total_frames + 1
    #frame = imutils.resize(frame, width=800)
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

        frame_resized = cv2.resize(rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)
    
        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std
    
        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()
    
        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
        #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
        

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
    
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                #cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                
                #tracker
                
               # rects.append((xmin,ymin, xmax,ymax))

                # compute the (x, y)-coordinates of the bounding box
                # for the object
                #(startX, startY, endX, endY) = boxes.astype("int")

                # construct a dlib rectangle object from the bounding
                # box coordinates and then start the dlib correlation
                # tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(xmin,ymin, xmax,ymax)
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
    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        print("objid", objectID)

        
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
                    to.counted = True

                # if the direction is positive (indicating the object
                # is moving down) AND the centroid is below the
                # center line, count the object
                elif direction > 0 and centroid[1] > H // 2:
                    totalDown += 1
                    to.counted = True

        # store the trackable object in our dictionary
        trackableObjects[objectID] = to

        #frame time ----end--------
    
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        #text = "ID: {}".format(objectID)
        text = "id:{},ft:{}sec".format(objectID, int(dwell_time[objectID]))
        cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        

    
    #count current perosn count and total person count
    lpc_count = len(objects)
    opc_count = len(object_id_list)

    lpc_txt = "LPC: {}".format(lpc_count)
    opc_txt = "OPC: {}".format(opc_count)

    cv2.putText(frame, lpc_txt, (5, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    cv2.putText(frame, opc_txt, (5, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    
    #fps
    
    fps_end_time = datetime.datetime.now()
    time_diff = fps_end_time - fps_start_time
    if time_diff.seconds == 0:
        fps = 0.0
    else:
        fps = (total_frames / time_diff.seconds)

    fps_text = "FPS: {:.2f}".format(fps)

    cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        
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
cv2.destroyAllWindows()