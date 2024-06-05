import cv2
import numpy as np
import setting
from surround_view import utils
from fisheye import Fisheye
import os,time
from panorama import Panorama
import threading
import queue
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict

def init(names,paramsfile,images,weightsfile,maskfile,is_detect) -> tuple[Fisheye,Panorama]:
    fisheyes = [Fisheye(p, n) for p,n in zip(paramsfile, names)]
    cys = []
    for i in range(4):
        fisheyes[i].build_one_map()
        cy = fisheyes[i].warpone(cv2.imread(images[i]))
        cys.append(cy)
    panorama = Panorama(cys,is_detect)
    panorama.imagespath =  [os.path.join("und_smimages", name + ".png") for name in names]
    panorama.load_weights_and_masks(weightsfile, maskfile)

    for i in fisheyes:
        i.start()
    panorama.fisheyes = fisheyes
    panorama.runmergethread()
    panorama.start()
    return fisheyes,panorama,images


if __name__ == "__main__":
    print("init")
    names = ['front','back', 'left', 'right']
    paramsfile = [os.path.join("./TankPanorama/my_yaml", name + ".yaml") for name in names]
    images = [os.path.join("./TankPanorama/und_smimages", name + ".png") for name in names]
    weightsfile = "weights.png"
    maskfile = "masks.png"
    fisheyes,panorama,images = init(names,paramsfile,images,weightsfile,maskfile,True)
    print("init finish")
    decs = []
    while True:
        last = time.time()
        for (f,img) in zip(fisheyes,images):
            f.queue_in.put(img)
        image = panorama.buffer.get()
        if panorama.detection.out_queue.full():
            decs = panorama.detection.out_queue.get()
        print(decs)
        if decs:
            image = panorama.detection.draw(image, decs)
            
        cv2.imshow("panorama", cv2.resize(image, None, fx=0.7, fy=0.7))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("close")
            for i in fisheyes:
                i.stopflag = True
            panorama.stopflag = True
            panorama.stopmergethread()
            break
        print("FPS", round(1/ (time.time() - last), 1))

    cv2.destroyAllWindows()
