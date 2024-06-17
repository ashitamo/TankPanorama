import cv2
from fisheye import Fisheye
import os,time
from panorama import Panorama


def init(names,paramsfile,images,weightsfile,maskfile) -> tuple[Fisheye,Panorama]:
    fisheyes = [Fisheye(p, n) for p,n in zip(paramsfile, names)]
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
    fisheyes,panorama,images = init(names,paramsfile,images,weightsfile,maskfile)
    print("init finish")
    decs = []
    while True:
        last = time.time()
        for (f,img) in zip(fisheyes,images):
            f.queue_in.put(img)
        image = panorama.buffer.get()
        cv2.imshow("panorama", cv2.resize(image, None, fx=0.7, fy=0.7))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("close")
            for i in fisheyes:
                i.stopflag = True
            panorama.stopflag = True
            panorama.stopmergethread()
            break
        print("FPS", round(1/ (time.time() - last), 1))

    cv2.destroyAllWindows()
