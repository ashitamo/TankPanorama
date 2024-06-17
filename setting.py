import os
import cv2
import numpy as np

camera_names = ["front", "back", "left", "right"]
'''
73*50 cm
'''
sph_foc_len = 410# 統一球體半徑

sfovx = 90 #每個相機裁切的fov x 方向大小
sfovy = 70 #每個相機裁切的fov y 方向大小

targetw = 1800 # 去畸變換後的寬
targeth = 700 # 去畸變換後的高

f = 238# 大約鏡頭焦距
patternsize = 70 # 真實基板高 (cm)
bodw =  1000 # 基板寬
bodh = 180 # 基板高
h1 = bodh*f/(targeth//2-bodh) # 真實基板高 (pixel)
pc_ratio = h1/patternsize # pixel : cm

cx,cy = (targetw/2, targeth/2)
albotw = (targeth-cy-bodh) * bodw / (targeth-cy)

bs = 25
flboth = ((bs+patternsize)*pc_ratio*targeth//2)/((bs+patternsize)*pc_ratio+f)
flbotw = (targeth-cy-(flboth)) * bodw / (targeth-cy)

flbodh = (bs*pc_ratio*targeth//2)/(bs*pc_ratio+f)
flbodw = (targeth-cy-flbodh) * bodw / (targeth-cy)

align_project_keypoing = {
    "front":[
        [(targetw-albotw)/2,targeth-bodh],
        [(targetw+albotw)/2,targeth-bodh],
        [(targetw-bodw)/2,targeth],
        [(targetw+bodw)/2,targeth]
    ],
    "back":[
        [(targetw-flbotw)/2,targeth-flboth],
        [(targetw+flbotw)/2,targeth-flboth],
        [(targetw-flbodw)/2,targeth-flbodh],
        [(targetw+flbodw)/2,targeth-flbodh]
    ],
    "left":[
        [(targetw-albotw)/2,targeth-bodh],
        [(targetw+albotw)/2,targeth-bodh],
        [(targetw-bodw)/2,targeth],
        [(targetw+bodw)/2,targeth]
    ],
    "right":[
        [(targetw-albotw)/2,targeth-bodh],
        [(targetw+albotw)/2,targeth-bodh],
        [(targetw-bodw)/2,targeth],
        [(targetw+bodw)/2,targeth]
    ],
}

affine_keypoint = {
    "front":[
        [targetw/2,targeth/2],
        [(targetw-flbotw)/2,targeth-flboth],
        [(targetw+flbotw)/2,targeth-flboth],
    ],
    "back":[
        [targetw/2,targeth/2],
        [(targetw-flbotw)/2,targeth-flboth],
        [(targetw+flbotw)/2,targeth-flboth],
    ]
}

if __name__ == "__main__":
    print(align_project_keypoing["front"])
    print(sph_foc_len/pc_ratio)
    im = np.ones((targeth, targetw,3), dtype=np.uint8)
    cv2.line(im, (int(align_project_keypoing["front"][0][0]), int(align_project_keypoing["front"][0][1])), (int(align_project_keypoing["front"][1][0]), int(align_project_keypoing["front"][1][1])), (255, 0, 0), 5)
    cv2.line(im, (int(align_project_keypoing["front"][1][0]), int(align_project_keypoing["front"][1][1])), (int(align_project_keypoing["front"][2][0]), int(align_project_keypoing["front"][2][1])), (255, 0, 0), 5)
    cv2.line(im, (int(align_project_keypoing["front"][2][0]), int(align_project_keypoing["front"][2][1])), (int(align_project_keypoing["front"][3][0]), int(align_project_keypoing["front"][3][1])), (255, 0, 0), 5)
    cv2.line(im, (int(align_project_keypoing["front"][3][0]), int(align_project_keypoing["front"][3][1])), (int(align_project_keypoing["front"][0][0]), int(align_project_keypoing["front"][0][1])), (255, 0, 0), 5)
    
    cv2.line(im, (int(align_project_keypoing["right"][0][0]), int(align_project_keypoing["right"][0][1])), (int(align_project_keypoing["right"][1][0]), int(align_project_keypoing["right"][1][1])), (255, 0, 0), 5)
    cv2.line(im, (int(align_project_keypoing["right"][1][0]), int(align_project_keypoing["right"][1][1])), (int(align_project_keypoing["right"][2][0]), int(align_project_keypoing["right"][2][1])), (255, 0, 0), 5)
    cv2.line(im, (int(align_project_keypoing["right"][2][0]), int(align_project_keypoing["right"][2][1])), (int(align_project_keypoing["right"][3][0]), int(align_project_keypoing["right"][3][1])), (255, 0, 0), 5)
    cv2.line(im, (int(align_project_keypoing["right"][3][0]), int(align_project_keypoing["right"][3][1])), (int(align_project_keypoing["right"][0][0]), int(align_project_keypoing["right"][0][1])), (255, 0, 0), 5)
    
    cv2.line(im, (int(align_project_keypoing["front"][0][0]), int(align_project_keypoing["front"][0][1])), (int(align_project_keypoing["right"][2][0]), int(align_project_keypoing["right"][2][1])), (255, 0, 0), 5)

    cv2.imshow("im", cv2.resize(im, None, fx=0.7, fy=0.7))
    cv2.waitKey(0)
# --------------------------------------------------------------------
# (shift_width, shift_height): how far away the birdview looks outside
# of the calibration pattern in horizontal and vertical directions
shift_w = 300
shift_h = 300

# size of the gap between the calibration pattern and the car
# in horizontal and vertical directions
inn_shift_w = 20
inn_shift_h = 50

# total width/height of the stitched image
total_w = 600 + 2 * shift_w
total_h = 1000 + 2 * shift_h

# four corners of the rectangular region occupied by the car
# top-left (x_left, y_top), bottom-right (x_right, y_bottom)
xl = shift_w + 180 + inn_shift_w
xr = total_w - xl
yt = shift_h + 200 + inn_shift_h
yb = total_h - yt
# --------------------------------------------------------------------

project_shapes = {
    "front": (total_w, yt),
    "back":  (total_w, yt),
    "left":  (total_h, xl),
    "right": (total_h, xl)
}

# pixel locations of the four points to be chosen.
# you must click these pixels in the same order when running
# the get_projection_map.py script
project_keypoints = {
    "front": [(shift_w + 120, shift_h),
              (shift_w + 480, shift_h),
              (shift_w + 120, shift_h + 160),
              (shift_w + 480, shift_h + 160)],

    "back":  [(shift_w + 120, shift_h),
              (shift_w + 480, shift_h),
              (shift_w + 120, shift_h + 160),
              (shift_w + 480, shift_h + 160)],

    "left":  [(shift_h + 360, shift_w),
              (shift_h + 680, shift_w),
              (shift_h + 360, shift_w + 160),
              (shift_h + 680, shift_w + 160)],

    "right": [(shift_h + 160, shift_w),
              (shift_h + 720, shift_w),
              (shift_h + 160, shift_w + 160),
              (shift_h + 720, shift_w + 160)]
}
# car_image = cv2.imread(os.path.join(os.getcwd(), "images", "car.png"))
# car_image = cv2.resize(car_image, (xr - xl, yb - yt))

def create_mesh_image(rows, cols, cell_size, line_color=(255, 255, 255), line_thickness=4):
    # 创建空白图像
    image = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8)

    # 绘制水平线
    for i in range(1, rows):
        y = i * cell_size
        cv2.line(image, (0, y), (cols * cell_size, y), line_color, line_thickness)

    # 绘制垂直线
    for j in range(1, cols):
        x = j * cell_size
        cv2.line(image, (x, 0), (x, rows * cell_size), line_color, line_thickness)

    return image


'''
camera_names = ["front", "back", "left", "right"]

sph_foc_len = 430# 統一球體半徑

targetw = 1500 # 去畸變換後的寬
targeth = 1000 # 去畸變換後的高

f = 254.8 # 大約鏡頭焦距
patternsize = 21 # 真實基板高 (cm)
bodw =  370 # 基板寬
bodh = 250 # 基板高
h1 = bodh*f/(targetw//2-bodh) # 真實基板高 (pixel)
pc_ratio = h1/patternsize # pixel : cm

cx,cy = (targetw/2, targeth/2)
albotw = (targeth-cy-bodh) * bodw / (targeth-cy)

align_project_keypoing = {
    "front":[
        [(targetw-albotw)/2,targeth-bodh],
        [(targetw+albotw)/2,targeth-bodh],
        [(targetw-bodw)/2,targeth],
        [(targetw+bodw)/2,targeth]
    ],
    "back":[
        [(targetw-albotw)/2,targeth-bodh],
        [(targetw+albotw)/2,targeth-bodh],
        [(targetw-bodw)/2,targeth],
        [(targetw+bodw)/2,targeth]
    ],
    "left":[
        [(targetw-albotw)/2,targeth-bodh],
        [(targetw+albotw)/2,targeth-bodh],
        [(targetw-bodw)/2,targeth],
        [(targetw+bodw)/2,targeth]
    ],
    "right":[
        [(targetw-albotw)/2,targeth-bodh],
        [(targetw+albotw)/2,targeth-bodh],
        [(targetw-bodw)/2,targeth],
        [(targetw+bodw)/2,targeth]
    ],
}'''