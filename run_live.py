import cv2
import numpy as np
import setting
from surround_view import utils
from fisheye import Fisheye
import os,time
from panorama import Panorama
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

def init(names,paramsfile,images,weightsfile,maskfile):
    # fisheyes = [Fisheye(p, n) for p,n in zip(paramsfile, names)]
    fisheyes = [None,None,None,None]
    fisheyes[0] = Fisheye(paramsfile[0], names[0],2)
    print('fisheye0 ok')
    fisheyes[1] = Fisheye(paramsfile[1], names[1],0)
    print('fisheye1 ok')
    fisheyes[2] = Fisheye(paramsfile[2], names[2],4)
    print('fisheye2 ok')
    fisheyes[3] = Fisheye(paramsfile[3], names[3],6)
    print('fisheye3 ok')
    cys = []
    for i in range(4):
        fisheyes[i].build_one_map()
        cy = fisheyes[i].warpone(cv2.imread(images[i]))
        cys.append(cy)
    panorama = Panorama(cys)
    panorama.imagespath =  [os.path.join("und_smimages", name + ".png") for name in names]
    panorama.load_weights_and_masks(weightsfile, maskfile)

    for i in fisheyes:
        i.start()
    panorama.fisheyes = fisheyes
    panorama.runmergethread()
    panorama.start()
    return fisheyes,panorama,images

if __name__ == "__main__":

    print("init")
    names = ['front','back', 'left', 'right']
    paramsfile = [os.path.join("./my_yaml", name + ".yaml") for name in names]
    images = [os.path.join("./und_smimages", name + ".png") for name in names]
    weightsfile = "weights.png"
    maskfile = "masks.png"
    fisheyes,panorama,_ = init(names,paramsfile,images,weightsfile,maskfile)
    panorama.stitch_all_parts()
    panorama.copyMakeBorder()
    print("init finish")
    writer = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 24, (2000,1000))


    # ex = -2
    # target = 85
    # def lightness(image):
    #     global ex
    #     gray_image = cv2.cvtColor(cv2.resize(image, (256,256)), cv2.COLOR_BGR2GRAY)
    #     average_brightness = cv2.mean(gray_image)[0]
    #     err = (target - average_brightness)
    #     ex = ex + 0.001*err
    #     print(ex)
    # for i in fisheyes:
    #     i.cap.set(cv2.CAP_PROP_EXPOSURE, ex)
    FPS = []
    while True:
        last = time.time()
        
        image = panorama.buffer.get()
        writer.write(cv2.resize(image,(2000,1000)))
        cv2.imshow("panorama", cv2.resize(image, None, fx=0.7, fy=0.7))
        
        # lightness(image)
        # print(ex)    
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            
            for i in fisheyes:
                i.stopflag = True
            panorama.stopflag = True
            panorama.stopmergethread()
            writer.release()
            print("close")
            break
        FPS.append(1/ (time.time() - last))
        if len(FPS)>5:
            del FPS[0]
        print("FPS", round(sum(FPS)/5, 1))

    cv2.destroyAllWindows()

