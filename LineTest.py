"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Manually select points to get the projection map
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import argparse
import os
import numpy as np
import cv2
from surround_view import PointSelector
import setting
from fisheye import Fisheye



def getLine(start,end,s=600):
    l = end - start
    l = l / np.linalg.norm(l)
    ss = np.arange(0, s, 100)
    ss = np.vstack((ss, ss)).T
    l = ss*l
    l[:,0] = l[:,0] + start[0]
    l[:,1] = l[:,1] + start[1]
    #cv2.polylines(image, [l.astype(np.int32)], False, (255, 0, 0), 5)
    return l

def drawForwadLine(image,x,y,camera,t=5):
    x = x-camera.NEW_K[0,2]
    y = y-camera.NEW_K[1,2]
    z = np.ones_like(x)
    z = z*setting.sph_foc_len
    r = np.sqrt(x**2+y**2+z**2)
    phi = np.arctan(x/z)
    theta = np.arctan(y/r)
    p = np.array((phi,theta,r)).T.reshape(-1,3)
    p = p*setting.sph_foc_len
    p = p+np.array([camera.NEW_K[0,2],camera.NEW_K[1,2],0]).reshape(-1,3)
    p = p[:,:2]-np.array((camera.ROI_x[0],camera.ROI_y[0])).reshape(-1,2)
    cv2.polylines(image, [p.astype(np.int32)], False, (0, 255, 0), t)

def main():
    name = "front"
    shift = (0, 0)
    camera_name = name
    camera_file = "my_yaml/" + name + ".yaml"
    image_file = "und_smimages/" + name + ".png"
    image = cv2.imread(image_file)
    camera = Fisheye(camera_file, camera_name)
    camera.NEW_K = camera.K.copy()
    camera.NEW_K[0,2] = camera.K[0,2]/640*setting.targetw
    camera.NEW_K[1,2] = camera.K[1,2]/480*setting.targeth
    #camera.K  = np.array([[0.625, 0, 0], [0, 0.625, 0], [0, 0, 1]]) @ camera.K 
    #camera.initnewK()
    #camera.set_shift(shift)
    camera.build_undistort_map()
    camera.build_align_map()
    camera.build_spherical_map()
    camera.build_one_map()

    foc_len = (camera.shp_K[0][0] + camera.shp_K[1][1])/2
    line = image.copy()
    line = cv2.remap(line, camera.map_x, camera.map_y, cv2.INTER_LINEAR)
    line = cv2.remap(line, camera.align_map_x, camera.align_map_y, cv2.INTER_LINEAR)
    print(200*setting.pc_ratio)
    l = getLine(
        np.array([600, int(setting.targeth)+250]),
        np.array([setting.cx, setting.cy])
    )
    rl = getLine(
        np.array([int(setting.targetw)-600, int(setting.targeth)+250]),
        np.array([setting.cx, setting.cy])
    )
    cv2.polylines(line,[l.astype(np.int32)], False, (255, 0, 0), 5)
    cv2.polylines(line,[rl.astype(np.int32)], False, (255, 0, 0), 5)
    cv2.imshow('rect',cv2.resize(line,None,fx=0.5,fy=0.5))

    image = camera.warpone(image)
    x,y = l[:,0],l[:,1]
    rx,ry = rl[:,0],rl[:,1]
    drawForwadLine(image,x,y,camera)
    drawForwadLine(image,rx,ry,camera) 
    cv2.imshow('fin',cv2.resize(image,None,fx=0.7,fy=0.7))
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
