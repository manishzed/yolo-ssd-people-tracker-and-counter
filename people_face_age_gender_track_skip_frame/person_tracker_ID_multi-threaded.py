import cv2
import datetime
import imutils
import numpy as np
from centroidtracker import CentroidTracker


#multi-threading---start-------
import cv2
import threading

class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
    def run(self):
        print("Starting " + self.previewName)
        camPreview(self.previewName, self.camID)

def camPreview(previewName, camID):
    protopath = r"E:\softweb\Pretrained_model\object_detectiom_model\MobileNetSSD_deploy.prototxt"
    modelpath = r"E:\softweb\Pretrained_model\object_detectiom_model\MobileNetSSD_deploy.caffemodel"
    detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
    
    # Only enable it if you are using OpenVino environment
    # detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
    # detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]
    
    tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)
    
    
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
    
    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0
    
    cv2.namedWindow(previewName)
    cam = cv2.VideoCapture(camID)
    if cam.isOpened():
        rval, frame = cam.read()
    else:
        rval = False

    while rval:
        
        rval, frame = cam.read()

        frame = imutils.resize(frame, width=600)
        total_frames = total_frames + 1
    
        (H, W) = frame.shape[:2]
    
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
    
        detector.setInput(blob)
        person_detections = detector.forward()
        rects = []
        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])
    
                if CLASSES[idx] != "person":
                    continue
    
                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                rects.append(person_box)
    
        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.3)
    
        objects = tracker.update(rects)
        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
    
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = "ID: {}".format(objectId)
            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    
        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)
    
        fps_text = "FPS: {:.2f}".format(fps)
    
        cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    
        #cv2.imshow("Application", frame)
        cv2.imshow(previewName, frame)
        # key = cv2.waitKey(20)
        # if key == 27:  # exit on ESC
        #     break
        # press "Q" to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

# Create threads as follows
video_path_1 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\motor_bike.mp4"
video_path_2 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\VPD_part2.mp4"
video_path_3 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\VPD_part1.mp4"


thread1 = camThread("Camera 1", video_path_1)
thread2 = camThread("Camera 2", video_path_2)
thread3 = camThread("Camera 3", video_path_3)


thread1.start()
thread2.start()
thread3.start()

print()
print("Active threads", threading.activeCount())