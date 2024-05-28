import cv2
import numpy as np
from fisheye import Fisheye

rotXdeg = 90
rotYdeg = 90
rotZdeg = 90
f = 256
dist = 256

def onRotXChange(val):
    global rotXdeg
    rotXdeg = val
def onRotYChange(val):
    global rotYdeg
    rotYdeg = val
def onRotZChange(val):
    global rotZdeg
    rotZdeg = val
def onFchange(val):
    global f
    f=val
def onDistChange(val):
    global dist
    dist=val
x = 800
def xchange(val):
    global x
    x = val-800
y = 800
def ychange(val):
    global y
    y = val-800

if __name__ == '__main__':

    fisheye = Fisheye("my_yaml/front.yaml","front")
    fisheye.build_undistort_map()
    src = fisheye.undistort(cv2.imread("und_smimages/front.png"))
    src = cv2.resize(src, None, fx=0.5, fy=0.5)
    print(fisheye.NEW_K)
    dst = np.ndarray(shape=src.shape,dtype=src.dtype)

    #Create user interface with trackbars that will allow to modify the parameters of the transformation
    wndname1 = "Source:"
    wndname2 = "WarpPerspective: "
    cv2.namedWindow(wndname1, 1)
    cv2.namedWindow(wndname2, 1)
    cv2.createTrackbar("Rotation X", wndname2, rotXdeg, 180, onRotXChange)
    cv2.createTrackbar("Rotation Y", wndname2, rotYdeg, 180, onRotYChange)
    cv2.createTrackbar("Rotation Z", wndname2, rotZdeg, 180, onRotZChange)
    cv2.createTrackbar("f", wndname2, f, 2000, onFchange)
    cv2.createTrackbar("Distance", wndname2, dist, 2000, onDistChange)
    cv2.createTrackbar("x", wndname2, x, 1600, xchange)
    cv2.createTrackbar("y", wndname2, y, 1600, ychange)

    #Show original image
    cv2.imshow(wndname1, src)

    h , w = src.shape[:2]

    while True:

        rotX = (rotXdeg - 90)*np.pi/180
        rotY = (rotYdeg - 90)*np.pi/180
        rotZ = (rotZdeg - 90)*np.pi/180

        #Projection 2D -> 3D matrix
        A1= np.matrix([[1, 0, -w/2],
                       [0, 1, -h/2],
                       [0, 0, 0   ],
                       [0, 0, 1   ]])

        # Rotation matrices around the X,Y,Z axis
        RX = np.matrix([[1,           0,            0, 0],
                        [0,np.cos(rotX),-np.sin(rotX), 0],
                        [0,np.sin(rotX),np.cos(rotX) , 0],
                        [0,           0,            0, 1]])

        RY = np.matrix([[ np.cos(rotY), 0, np.sin(rotY), 0],
                        [            0, 1,            0, 0],
                        [ -np.sin(rotY), 0, np.cos(rotY), 0],
                        [            0, 0,            0, 1]])

        RZ = np.matrix([[ np.cos(rotZ), -np.sin(rotZ), 0, 0],
                        [ np.sin(rotZ), np.cos(rotZ), 0, 0],
                        [            0,            0, 1, 0],
                        [            0,            0, 0, 1]])

        #Composed rotation matrix with (RX,RY,RZ)
        R = RX * RY * RZ

        #Translation matrix on the Z axis change dist will change the height
        T = np.matrix([[1,0,0,x],
                       [0,1,0,y],
                       [0,0,1,dist],
                       [0,0,0,1]])

        #Camera Intrisecs matrix 3D -> 2D
        A2 = np.matrix([[f, 0, w/2,0],
                       [0, f, h/2,0],
                       [0, 0,   1,0]])

        # Final and overall transformation matrix
        H = A2 @ R @ T @ A1
        print(T @ R)
        # Apply matrix transformation
        cv2.warpPerspective(src, H, (w, h), dst, cv2.INTER_CUBIC)
        cv2.circle(dst, (w//2, h//2), 3, (0, 0, 255), -1)
        #Show the image
        cv2.imshow(wndname2, dst)
        cv2.waitKey(1)