
import cv2
import numpy as np
from surround_view import utils

def white_balance(image):
    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def mm(original):
    # 灰度图
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)
    gray_ = cv2.equalizeHist(gray)
    cv2.imshow("gray_", gray_)

    # YUV :亮度 色度 饱和度
    yuv = cv2.cvtColor(original, cv2.COLOR_BGR2YUV)
    yuv[..., 0] = cv2.equalizeHist(yuv[..., 0])
    equalized_color = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    cv2.imshow("equalized_color", equalized_color)
    return equalized_color

cap = cv2.VideoCapture(r'C:\Users\hsun9\Desktop\TankVision\\未命名.mp4')


while True:
    ret, frame = cap.read()
    if ret == False:
        break
    #frame = white_balance(frame)
    #frame = utils.make_white_balance(frame)
    #frame = mm(frame)
    cv2.imshow('frame', cv2.resize(frame, None, fx=0.8, fy=0.8))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break