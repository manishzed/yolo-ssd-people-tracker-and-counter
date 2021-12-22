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
    cv2.namedWindow(previewName)
    cam = cv2.VideoCapture(camID)
    if cam.isOpened():
        rval, frame = cam.read()
    else:
        rval = False

    while rval:
        
        rval, frame = cam.read()
        if rval:
            cv2.imshow(previewName, frame)
            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                break
    cv2.destroyWindow(previewName)

# Create threads as follows
video_path_1 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\motor_bike.mp4"
video_path_2 =r"C:\Users\hp\Desktop\priyasoftweb\person_tracker_id_frame_time\input\VPD_part2.mp4"


thread1 = camThread("Camera 1", video_path_1)
thread2 = camThread("Camera 2", video_path_2)


thread1.start()
thread2.start()

print()
print("Active threads", threading.activeCount())