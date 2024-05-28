"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Manually select points to get the projection map
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
import argparse
import os
import numpy as np
import cv2
from fisheye import Fisheye
from surround_view import PointSelector

import setting


def get_projection_map(camera_model, image):
    scale = 0.5
    und_image = camera_model.undistort(image)
    align_image = camera_model.aling_project(und_image)
    name = camera_model.camera_name
    gui = PointSelector(cv2.resize(align_image, None, fx=scale, fy=scale), title=name)
    dst_points = setting.project_keypoints[name]
    choice = gui.loop()
    if choice > 0:
        src = np.float32(gui.keypoints)/scale
        dst = np.float32(dst_points)
        camera_model.project_matrix = cv2.getPerspectiveTransform(src, dst)
        proj_image = cv2.warpPerspective(align_image, camera_model.project_matrix, (setting.project_shapes[name][0], setting.project_shapes[name][1]))
        cv2.imshow(name, cv2.resize(proj_image, None, fx=0.5, fy=0.5))
        cv2.waitKey(0)
        return True
    return False


def main():
    name = "back"
    camera_name = name
    camera_file = "yaml/" + name + ".yaml"
    image_file = "images/" + name + ".png"
    image = cv2.imread(image_file)
    camera = Fisheye(camera_file, camera_name)
    success = get_projection_map(camera, image)
    if success:
        print("saving projection matrix to yaml")
        #camera.save_data()
    else:
        print("failed to compute the projection map")


if __name__ == "__main__":
    main()
