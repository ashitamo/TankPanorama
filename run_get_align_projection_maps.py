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


def get_projection_map(camera_model, image,name):
    und_image = image
    und_image = camera_model.undistort(image)
    print(und_image.shape)
    und_image = cv2.circle(und_image,(int(und_image.shape[1]//2),int(und_image.shape[0]//2)), 10, (0,0,255), 10)
    name = camera_model.camera_name
    scale = 0.7
    gui = PointSelector(cv2.resize(und_image, None, fx=scale, fy=scale), title=name)
    choice = gui.loop()
    dst_points = setting.align_project_keypoing[name]
    if choice > 0:
        src = np.float32(gui.keypoints) / scale
        print(src)
        print(src[3][0]-src[2][0])
        print(src[2][1]-src[0][1])
        dst = np.float32(dst_points)
        camera_model.align_project_matrix = cv2.getPerspectiveTransform(src, dst)
        camera_model.build_align_map()
        proj_image = camera_model.aling_project(und_image)
        proj_image = cv2.circle(proj_image,(int(setting.cx), int(setting.cy)), 10, (0,0,255), 10)
        cv2.line(proj_image, (int(setting.align_project_keypoing["front"][0][0]), int(setting.align_project_keypoing["front"][0][1])), (int(setting.align_project_keypoing["front"][1][0]), int(setting.align_project_keypoing["front"][1][1])), (255, 0, 0), 5)
        cv2.line(proj_image, (int(setting.align_project_keypoing["front"][1][0]), int(setting.align_project_keypoing["front"][1][1])), (int(setting.align_project_keypoing["front"][2][0]), int(setting.align_project_keypoing["front"][2][1])), (255, 0, 0), 5)
        cv2.line(proj_image, (int(setting.align_project_keypoing["front"][2][0]), int(setting.align_project_keypoing["front"][2][1])), (int(setting.align_project_keypoing["front"][3][0]), int(setting.align_project_keypoing["front"][3][1])), (255, 0, 0), 5)
        cv2.line(proj_image, (int(setting.align_project_keypoing["front"][3][0]), int(setting.align_project_keypoing["front"][3][1])), (int(setting.align_project_keypoing["front"][0][0]), int(setting.align_project_keypoing["front"][0][1])), (255, 0, 0), 5)

        cv2.imshow(name, cv2.resize(proj_image, None, fx=scale, fy=scale))
        # smimg = cv2.resize(proj_image, None, fx=0.8, fy=0.5)
        # smimg = cv2.copyMakeBorder(smimg,int(proj_image.shape[0]-smimg.shape[0])//2,int(proj_image.shape[0]-smimg.shape[0])//2,int(proj_image.shape[1]-smimg.shape[1])//2,int(proj_image.shape[1]-smimg.shape[1])//2,cv2.BORDER_CONSTANT,value=(0,0,0))
        # cv2.line(smimg, (int(setting.align_project_keypoing["back"][0][0]), int(setting.align_project_keypoing["back"][0][1])), (int(setting.align_project_keypoing["back"][1][0]), int(setting.align_project_keypoing["back"][1][1])), (255, 0, 0), 5)
        # cv2.line(smimg, (int(setting.align_project_keypoing["back"][1][0]), int(setting.align_project_keypoing["back"][1][1])), (int(setting.align_project_keypoing["back"][2][0]), int(setting.align_project_keypoing["back"][2][1])), (255, 0, 0), 5)
        # cv2.line(smimg, (int(setting.align_project_keypoing["back"][2][0]), int(setting.align_project_keypoing["back"][2][1])), (int(setting.align_project_keypoing["back"][3][0]), int(setting.align_project_keypoing["back"][3][1])), (255, 0, 0), 5)
        # cv2.line(smimg, (int(setting.align_project_keypoing["back"][3][0]), int(setting.align_project_keypoing["back"][3][1])), (int(setting.align_project_keypoing["back"][0][0]), int(setting.align_project_keypoing["back"][0][1])), (255, 0, 0), 5)

        # cv2.imshow('wdwd',cv2.resize(smimg,None,fx=0.3,fy=0.3))

        print(proj_image.shape)
        cv2.waitKey(0)
        return True
    return False



def main():
    name = "back"
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
    print(camera.NEW_K)
    success = get_projection_map(camera, image,name)
    if success:
        print("saving projection matrix to yaml")
        camera.save_data()
    else:
        print("failed to compute the projection map")


if __name__ == "__main__":
    main()
