import cv2
import threading
import queue
import os
import time
from ultralytics import YOLO
import random

class Detection(threading.Thread):
    def __init__(self,weight_path):
        super().__init__()
        self.daemon = True
        self.model = YOLO(weight_path)
        self.stopflag = False
        self.in_queue = queue.Queue(2)
        self.out_queue = queue.Queue(2)

    def boxes2AngleDistance(self,results):
        decs = []
        xyxyn = results[0].boxes.xyxyn.cpu().numpy()
        xyxyn[:,[0,2]] = xyxyn[:,[0,2]]*360
        xyxyn[:,[1,3]] = xyxyn[:,[1,3]]*180
        cls = results[0].boxes.cls.cpu().numpy().tolist()
        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.cpu().numpy().tolist()
        else:
            track_ids = []
        for i in zip(xyxyn,cls,track_ids):
            decdict = {
                "lefttop": [i[0][0],i[0][1]],
                "rightbottom": [i[0][2],i[0][3]],
                "class": i[1],
                "track_id": i[2]
            }
            decs.append(decdict)
        return decs

    def draw(self, image, decs):
        for dec in decs:
            xyxy = dec["lefttop"] + dec["rightbottom"]
            deg = (xyxy[2] + xyxy[0])/2
            xyxy[0] = xyxy[0]/360 * image.shape[1]
            xyxy[1] = xyxy[1]/180 * image.shape[0]
            xyxy[2] = xyxy[2]/360 * image.shape[1]
            xyxy[3] = xyxy[3]/180 * image.shape[0]
            color = (128, 255, 0)
            cv2.rectangle(image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
            cv2.putText(image, "cls: " + str(int(dec["class"])), (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            cv2.putText(image, "deg: " + str(int(deg)), (int(xyxy[0]), int(xyxy[1])-40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            cv2.putText(image, "id: " + str(int(dec["track_id"])), (int(xyxy[0]), int(xyxy[1])-70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        return image
    def run(self):
        while not self.stopflag:
            try:
                image = self.in_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            results = self.model.track(image, persist=True, verbose=False,conf=0.3)
            decs = self.boxes2AngleDistance(results)
            if not self.out_queue.full():
                self.out_queue.put(decs, timeout=0.1)

class ParanomaReceiver(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.stopflag = False
        self.out_queue = queue.Queue(1)
        self.cap = cv2.VideoCapture("output1.mp4")

    def copyMakeBorder(self,image):
        padding = int( (image.shape[1]/2 - image.shape[0]) // 2)
        image = cv2.copyMakeBorder(image, padding, padding, 0, 0, cv2.BORDER_CONSTANT,value=[0,0,0])
        return image

    def run(self):
        while not self.stopflag:
            image = self.cap.read()[1]
            self.out_queue.put(image)


if __name__ == "__main__":
    paranomaReceiver = ParanomaReceiver()
    paranomaReceiver.start()
    detection = Detection("yolov8n.pt")
    detection.start()
    decs = []
    while True:
        last = time.time()
        image = paranomaReceiver.out_queue.get()

        if not detection.in_queue.full():
            detection.in_queue.put(image)
        if not detection.out_queue.empty():
            decs = detection.out_queue.get()
        image = detection.draw(image, decs)
        
        cv2.imshow("image", cv2.resize(image, None, fx=0.7, fy=0.7))
        cv2.waitKey(1)
        sl = 0.06 - (time.time() - last)
        time.sleep(0 if sl < 0 else sl)
