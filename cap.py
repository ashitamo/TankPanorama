import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import time
import numpy as np
import pickle
import setting
import sys

number = 0
name = "left"
c = 4

if sys.platform == "win32":
    cam1 = cv2.VideoCapture(number, cv2.CAP_DSHOW)
else:
    cam1 = cv2.VideoCapture(number)

cam1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
desired_width = 640 # 你想設定的寬度
desired_height = 480  # 你想設定的高度
cam1.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
actual_width = cam1.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cam1.get(cv2.CAP_PROP_FRAME_HEIGHT)
# cam1.set(cv2.CAP_PROP_FORMAT,-1)
# cam1.set(cv2.CAP_PROP_CONVERT_RGB, 0)
print("實際設定的解析度:", (actual_width, actual_height))
desired_fps = 30  # 你想設定的FPS值
cam1.set(cv2.CAP_PROP_FPS, desired_fps)
# 檢查實際設定的FPS
actual_fps = cam1.get(cv2.CAP_PROP_FPS)
print("實際設定的FPS:", actual_fps)


fs = cv2.FileStorage(f"my_yaml/{name}.yaml", cv2.FILE_STORAGE_READ)
K = fs.getNode("camera_matrix").mat()
D = fs.getNode("dist_coeffs").mat()
fs.release()
print(K)
print(D)
new_K = K.copy()
new_K[0,2],new_K[1,2] = K[0,2]/640*setting.targetw, K[1,2]/480*setting.targeth

map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (setting.targetw, setting.targeth), cv2.CV_16SC2)


def undistort(img):
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


while True:
    cam1.grab()
    ret1, frame1o = cam1.retrieve()
    if ret1:
        cv2.imshow('frame1o', frame1o)
        frame1 = undistort(frame1o)
        cv2.line(frame1,(int(new_K[0,2]),int(new_K[1,2])-300),(int(new_K[0,2]),int(new_K[1,2]+300)),(0,0,255),5)
        cv2.line(frame1,(int(new_K[0,2]-100),int(new_K[1,2])),(int(new_K[0,2]+100),int(new_K[1,2])),(0,0,255),5)
        cv2.imshow('frame1', cv2.resize(frame1, None, fx=0.7, fy=0.7))
    k = cv2.waitKey(1)
    if k & 0xFF == ord(' '):
        print("saving frame")
        cv2.imwrite(f'und_smimages/frame{c}.png', frame1o)
        c+=1
    elif k & 0xFF == ord('q'):
        break
        