import cv2
import numpy as np
import setting
from surround_view import utils
from fisheye import Fisheye
import os,time
from panorama import Panorama
from PIL import Image

if __name__ == "__main__":
    names = ['front','back', 'left', 'right']
    paramsfile = [os.path.join("my_yaml", name + ".yaml") for name in names]
    images = [os.path.join("und_smimages", name + ".png") for name in names]
    fisheyes = [Fisheye(p, n) for p,n in zip(paramsfile, names)]
    for i in range(4):
        fisheyes[i].build_undistort_map()
        fisheyes[i].build_align_map()
    undimages = [fisheyes[i].undistort(cv2.imread(image)) for i, image in enumerate(images)]
    aligns = [fisheyes[i].aling_project(undimage) for i, undimage in enumerate(undimages)]
    for i in range(4):
        cv2.imshow(f"und{i}", cv2.resize(undimages[i], None, fx=0.5, fy=0.5))
    cys = []
    onecys = []
    for i in range(4):
        fisheyes[i].build_one_map()
        fisheyes[i].build_spherical_map()
        cy = fisheyes[i].warp_spherical(aligns[i])
        onecy = fisheyes[i].warpone(cv2.imread(images[i]))
        cys.append(cy)
        onecys.append(onecy)
        #cv2.imshow(f"cy{i}", cv2.resize(cy, None, fx=0.9, fy=0.9))
        cy = fisheyes[i].warp_spherical(aligns[i])
        cv2.imshow(f"ccy{i}", cv2.resize(cy, None, fx=0.9, fy=0.9))
    panorama = Panorama(onecys)

    G,M = panorama.get_weights_and_masks()
    cv2.waitKey(0)
    last = time.time()
    panorama.stitch_all_parts()
    def white_balance(img):
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result
    panorama.image = white_balance(panorama.image)

    #panorama.make_white_balance()
    cv2.imshow("panorama", cv2.resize(panorama.image, None, fx=0.8, fy=0.8))
    print(time.time() - last)
    print(panorama.image.shape)
    cv2.waitKey(0)
    panorama.updata_images(onecys)
    panorama.stitch_all_parts()
    panorama.make_white_balance()
    panorama.copyMakeBorder()
    cv2.imshow("panorama", cv2.resize(panorama.image, None, fx=0.9, fy=0.9))
    cv2.waitKey(0)
    cv2.imwrite("panorama.png", panorama.image)
    print("weights.png and masks.png saved")
    Image.fromarray((G * 255).astype(np.uint8)).save("weights.png")
    Image.fromarray(M.astype(np.uint8)).save("masks.png")
