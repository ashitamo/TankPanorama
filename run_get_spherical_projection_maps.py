import numpy as np
import cv2
from fisheye import Fisheye
import setting
from surround_view import utils

foc_lenx = setting.f
foc_leny = setting.f

'''
setting.sph_foc_len 要先設定
'''

name = "front"
yaml = "my_yaml/" + name + ".yaml"
image = "und_smimages/" + name + ".png"
fisheye = Fisheye(yaml, name)
image = cv2.imread(image)
fisheye.build_undistort_map()
fisheye.build_align_map()
undistorted = fisheye.undistort(image)
cv2.imshow("undistorted", cv2.resize(undistorted, None, fx=0.2, fy=0.2))
align = fisheye.aling_project(undistorted)
cv2.imshow("align", cv2.resize(align, None, fx=0.2, fy=0.2))
cv2.waitKey(1)


fisheye.shp_K = np.array([
    [foc_lenx , 0.00000000e+00, setting.cx],
    [0.00000000e+00, foc_leny , setting.cy],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])
fisheye.build_spherical_map()
cy = fisheye.warp_spherical(align)
cv2.imshow("cy", cv2.resize(cy, None, fx=0.5, fy=0.5))
cv2.waitKey(0)

fisheye.save_data()