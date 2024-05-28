import cv2
import numpy as np
import os
from fisheye import Fisheye
import setting

def warp_spherical(img,K):
    foc_len = (K[0][0] + K[1][1])/2
    temp = np.mgrid[0:img.shape[1],0:img.shape[0]]
    x,y = temp[0],temp[1]
    phi = (x - K[0][2])/foc_len # angle phi
    theta = (y - K[1][2])/foc_len # theta
    p = np.array([np.cos(theta) * np.sin(phi) , np.sin(theta),np.cos(theta)*np.cos(phi)]).T.reshape(-1,3)
    image_points = K.dot(p.T).T
    points = image_points[:,:-1]/image_points[:,[-1]]
    points = points.reshape(img.shape[0],img.shape[1],-1)
    cylinder = cv2.remap(img, (points[:, :, 0]).astype(np.float32), (points[:, :, 1]).astype(np.float32), cv2.INTER_LINEAR)
    return cylinder

names = ['front','back', 'left', 'right']
paramsfile = [os.path.join("my_yaml", name + ".yaml") for name in names]
images = [os.path.join("smimages", name + ".png") for name in names]
 

front_fish = Fisheye(paramsfile[0], names[0])
back_fish = Fisheye(paramsfile[1], names[1])
left_fish = Fisheye(paramsfile[2], names[2])
right_fish = Fisheye(paramsfile[3], names[3])


undimages = [front_fish.undistort(cv2.imread(image)) for image in images]
aligns = [front_fish.aling_project(undimage) for undimage in undimages]


cys = []
for i in range(4):
    
    cy = warp_spherical(aligns[i],np.array([
        [1000 , 0.00000000e+00, setting.cx],
        [0.00000000e+00, 1000 , setting.cy],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ]))
    # fov = 120
    cy = cy[int(setting.cy-1047):int(setting.cy+1047),int(setting.cx-1047):int(setting.cx+1047),:]
    cv2.imshow(names[i], cv2.resize(cy, None, fx=0.3, fy=0.3))
    cys.append(cy)

front_fish.get_weights_and_masks(cys)

cv2.waitKey(0)