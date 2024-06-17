import cv2
import numpy as np
import setting
from surround_view import utils
import threading
from PIL import Image
import queue
import os
import time
from fisheye import Fisheye

def merge(imA, imB, G):
    out = (np.multiply(imA, G) + np.multiply(imB, 1 - G)).astype(np.uint8)
    return out


class mergeThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(1)
        self.buffer = queue.Queue(1)
        self.stopflag = False
    def run(self):
        while not self.stopflag:
            try:
                imA, imB, G = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self.buffer.put(merge(imA, imB, G))



class Panorama(threading.Thread):
    def __init__(self,images):
        self.cx,self.cy = images[0].shape[1]//2,images[0].shape[0]//2

        self.OV_FOV = 20
        self.F_FOV = (30 , 30) #120 ->100 50,50
        self.L_FOV = (35 , 20) #110 -> 90 50,40  
        self.R_FOV = (20 , 35) #110 -> 90 40,50
        self.B_FOV = (15 , 15) #100 -> 80 40,40
        '''
        self.OV_FOV = 20
        self.F_FOV = (30 , 30) #120 ->100 50,50
        self.L_FOV = (30 , 20) #110 -> 90 50,40  
        self.R_FOV = (20 , 30) #110 -> 90 40,50
        self.B_FOV = (20 , 20) #100 -> 80 40,40
        '''

        self.ROI_F,self.ROI_F_OVL,self.ROI_F_OVR = utils.dfov_to_pixel(self.F_FOV[0],self.F_FOV[1],setting.sph_foc_len,self.cx,self.cy,self.OV_FOV)
        self.ROI_L,self.ROI_L_OVL,self.ROI_L_OVR = utils.dfov_to_pixel(self.L_FOV[0],self.L_FOV[1],setting.sph_foc_len,self.cx,self.cy,self.OV_FOV)
        self.ROI_R,self.ROI_R_OVL,self.ROI_R_OVR = utils.dfov_to_pixel(self.R_FOV[0],self.R_FOV[1],setting.sph_foc_len,self.cx,self.cy,self.OV_FOV)
        self.ROI_B,self.ROI_B_OVL,self.ROI_B_OVR = utils.dfov_to_pixel(self.B_FOV[0],self.B_FOV[1],setting.sph_foc_len,self.cx,self.cy,self.OV_FOV)
        

        self.images = images

        self.threads = []
        self.updata_images(images)


        self.W = (self.FL.shape[1] + self.F.shape[1]) * 4
        self.H = self.F.shape[0]
        self.image = np.zeros((self.H, self.W, 3), np.uint8)

        super().__init__()
        self.stopflag = False
        self.buffer = queue.Queue(1)
        self.fisheyes = None
        self.imagespath = None

    def load_weights_and_masks(self, weights_image, masks_image):
        GMat = np.asarray(Image.open(weights_image).convert("RGBA"), dtype=np.float64) / 255.0
        self.weights = [np.stack((GMat[:, :, k],
                                  GMat[:, :, k],
                                  GMat[:, :, k]), axis=2)
                        for k in range(4)]

        Mmat = np.asarray(Image.open(masks_image).convert("RGBA"), dtype=np.float64)
        Mmat = utils.convert_binary_to_bool(Mmat)
        self.masks = [Mmat[:, :, k] for k in range(4)]

    @property
    def FL(self):
        return self.images[0][:,self.ROI_F_OVL[0]:self.ROI_F_OVL[1]]

    @property
    def F(self):
        return self.images[0][:,self.ROI_F[0]:self.ROI_F[1]]

    @property
    def FR(self):
        return self.images[0][:,self.ROI_F_OVR[0]:self.ROI_F_OVR[1]]

    @property
    def BR(self):
        return self.images[1][:,self.ROI_B_OVL[0]:self.ROI_B_OVL[1]]
        
    @property
    def B(self):
        return self.images[1][:,self.ROI_B[0]:self.ROI_B[1]]

    @property
    def BL(self):
        return self.images[1][:,self.ROI_B_OVR[0]:self.ROI_B_OVR[1]]


    @property
    def LB(self):
        return self.images[2][:,self.ROI_L_OVL[0]:self.ROI_L_OVL[1]]

    @property
    def L(self):
        return self.images[2][:,self.ROI_L[0]:self.ROI_L[1]]

    @property
    def LF(self):
        return self.images[2][:,self.ROI_L_OVR[0]:self.ROI_L_OVR[1]]

    @property
    def RF(self):
        return self.images[3][:,self.ROI_R_OVL[0]:self.ROI_R_OVL[1]]

    @property
    def R(self):
        return self.images[3][:,self.ROI_R[0]:self.ROI_R[1]]

    @property
    def RB(self):
        return self.images[3][:,self.ROI_R_OVR[0]:self.ROI_R_OVR[1]]

    def updata_images(self,images):
        #del self.images
        self.images = images

    def get_weights_and_masks(self):
        # G0, M0 = utils.get_weight_mask_matrix(self.FL,self.LF)
        # G1, M1 = utils.get_weight_mask_matrix(self.FR,self.RF)
        # G2, M2 = utils.get_weight_mask_matrix(self.RB,self.BR)
        # G3, M3 = utils.get_weight_mask_matrix(self.BL,self.LB)

        G0, M0 = utils.get_weight_mask_matrix_by_board_dist(self.FL,self.LF)
        G1, M1 = utils.get_weight_mask_matrix_by_board_dist(self.RF,self.FR)
        G2, M2 = utils.get_weight_mask_matrix_by_board_dist(self.RB,self.BR,True)
        G3, M3 = utils.get_weight_mask_matrix_by_board_dist(self.BL,self.LB,True)

        cv2.imshow('M0',cv2.resize(M0,None,fx=0.9,fy=0.9))
        cv2.imshow('G0',cv2.resize(G0,None,fx=0.9,fy=0.9))
        cv2.imshow('M1',cv2.resize(M1,None,fx=0.9,fy=0.9))
        cv2.imshow('G1',cv2.resize(G1,None,fx=0.9,fy=0.9))
        cv2.imshow('M2',cv2.resize(M2,None,fx=0.9,fy=0.9))  
        cv2.imshow('G2',cv2.resize(G2,None,fx=0.9,fy=0.9))
        cv2.imshow('M3',cv2.resize(M3,None,fx=0.9,fy=0.9))
        cv2.imshow('G3',cv2.resize(G3,None,fx=0.9,fy=0.9))


        self.weights = [np.stack((G, G, G), axis=2) for G in (G0, G1, G2, G3)]
        self.masks = [(M / 255.0).astype(int) for M in (M0, M1, M2, M3)]
        return np.stack((G0, G1, G2, G3), axis=2), np.stack((M0, M1, M2, M3), axis=2)

    def runmergethread(self):
        self.threads = []
        self.threads.append(mergeThread())
        self.threads.append(mergeThread())
        self.threads.append(mergeThread())
        self.threads.append(mergeThread())
        for t in self.threads:
            t.start()

    def stopmergethread(self):
        for t in self.threads:
            t.stopflag = True
            
    def stitch_all_parts(self):
        if len(self.threads) != 0:
            self.threads[0].queue.put((self.FL, self.LF, self.weights[0]))
            self.threads[1].queue.put((self.RF, self.FR, self.weights[1]))
            self.threads[2].queue.put((self.RB, self.BR, self.weights[2]))
            self.threads[3].queue.put((self.BL, self.LB, self.weights[3]))
            out1 = self.threads[0].buffer.get()
            out2 = self.threads[1].buffer.get()
            out3 = self.threads[2].buffer.get()
            out4 = self.threads[3].buffer.get()
        else:
            out1 = merge(self.FL, self.LF, self.weights[0])
            out2 = merge(self.RF, self.FR, self.weights[1])
            out3 = merge(self.RB, self.BR, self.weights[2])
            out4 = merge(self.BL, self.LB, self.weights[3])
            # out1 = self.FL
            # out2 = self.FR
            # out3 = self.RB
            # out4 = self.LB


        self.image = np.hstack(
            [
                self.L,
                out1,
                self.F,
                out2,
                self.R,
                out3,
                self.B,
                out4
            ]
        )

        # left_bound = 0
        # right_bound = self.L.shape[1]
        # self.image[:,left_bound:right_bound] = self.L
        # left_bound = right_bound
        # right_bound += self.LF.shape[1]
        # self.image[:,left_bound:right_bound] = out1
        # left_bound = right_bound
        # right_bound += self.F.shape[1]
        # self.image[:,left_bound:right_bound] = self.F
        # left_bound = right_bound
        # right_bound += self.FR.shape[1]
        # self.image[:,left_bound:right_bound] = out2
        # left_bound = right_bound
        # right_bound += self.R.shape[1]
        # self.image[:,left_bound:right_bound] = self.R
        # left_bound = right_bound
        # right_bound += self.RB.shape[1]
        # self.image[:,left_bound:right_bound] = out3
        # left_bound = right_bound
        # right_bound += self.B.shape[1]
        # self.image[:,left_bound:right_bound] = self.B
        # left_bound = right_bound
        # right_bound += self.BL.shape[1]
        # self.image[:,left_bound:right_bound] = out4

    def copyMakeBorder(self):
        padding = int( (self.image.shape[1]/2 - self.image.shape[0]) // 2)
        self.image = cv2.copyMakeBorder(self.image, padding, padding, 0, 0, cv2.BORDER_CONSTANT,value=[0,0,0])
    def make_white_balance(self):
        self.image = utils.make_white_balance(self.image)

    def white_balance(self):
        result = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        self.image = result

    def run(self):
        while not self.stopflag:
            # 创建线程并启动
            result = [None,None,None,None]
            result[0] = self.fisheyes[0].queue_out.get()
            result[1] = self.fisheyes[1].queue_out.get()
            result[2] = self.fisheyes[2].queue_out.get()
            result[3] = self.fisheyes[3].queue_out.get()
            self.updata_images(result)
            self.stitch_all_parts()
            #self.make_white_balance()
            self.white_balance()
            self.copyMakeBorder()
            self.buffer.put(self.image)
            



    # def make_luminance_balance(self):

    #     def tune(x):
    #         if x >= 1:
    #             return x * np.exp((1 - x) * 0.5)
    #         else:
    #             return x * np.exp((1 - x) * 0.8)

    #     m1, m2, m3, m4 = self.masks

    #     a1 = utils.mean_luminance_ratio(FLb, LFb, m1)
    #     a2 = utils.mean_luminance_ratio(FLg, LFg, m1)
    #     a3 = utils.mean_luminance_ratio(FLr, LFr, m1)

    #     b1 = utils.mean_luminance_ratio(FRb, RFb, m2)
    #     b2 = utils.mean_luminance_ratio(FRg, RFg, m2)
    #     b3 = utils.mean_luminance_ratio(FRr, RFr, m2)

    #     c1 = utils.mean_luminance_ratio(RBb, BRb, m3)
    #     c2 = utils.mean_luminance_ratio(RBg, BRg, m3)
    #     c3 = utils.mean_luminance_ratio(RBr, BRr, m3)

    #     d1 = utils.mean_luminance_ratio(LBb, BLb, m4)
    #     d2 = utils.mean_luminance_ratio(LBg, BLg, m4)
    #     d3 = utils.mean_luminance_ratio(LBr, BLr, m4)

    #     t1 = (a1 * b1 * c1 * d1)**0.25
    #     t2 = (a2 * b2 * c2 * d2)**0.25
    #     t3 = (a3 * b3 * c3 * d3)**0.25

    #     x1 = t1 / (d1 / a1)**0.5
    #     x2 = t2 / (d2 / a2)**0.5
    #     x3 = t3 / (d3 / a3)**0.5

    #     x1 = tune(x1)
    #     x2 = tune(x2)
    #     x3 = tune(x3)

    #     Fb = utils.adjust_luminance(Fb, x1)
    #     Fg = utils.adjust_luminance(Fg, x2)
    #     Fr = utils.adjust_luminance(Fr, x3)

    #     y1 = t1 / (b1 / c1)**0.5
    #     y2 = t2 / (b2 / c2)**0.5
    #     y3 = t3 / (b3 / c3)**0.5

    #     y1 = tune(y1)
    #     y2 = tune(y2)
    #     y3 = tune(y3)

    #     Bb = utils.adjust_luminance(Bb, y1)
    #     Bg = utils.adjust_luminance(Bg, y2)
    #     Br = utils.adjust_luminance(Br, y3)

    #     z1 = t1 / (c1 / d1)**0.5
    #     z2 = t2 / (c2 / d2)**0.5
    #     z3 = t3 / (c3 / d3)**0.5

    #     z1 = tune(z1)
    #     z2 = tune(z2)
    #     z3 = tune(z3)

    #     Lb = utils.adjust_luminance(Lb, z1)
    #     Lg = utils.adjust_luminance(Lg, z2)
    #     Lr = utils.adjust_luminance(Lr, z3)

    #     w1 = t1 / (a1 / b1)**0.5
    #     w2 = t2 / (a2 / b2)**0.5
    #     w3 = t3 / (a3 / b3)**0.5

    #     w1 = tune(w1)
    #     w2 = tune(w2)
    #     w3 = tune(w3)

    #     Rb = utils.adjust_luminance(Rb, w1)
    #     Rg = utils.adjust_luminance(Rg, w2)
    #     Rr = utils.adjust_luminance(Rr, w3)

    #     self.frames = [cv2.merge((Fb, Fg, Fr)),
    #                    cv2.merge((Bb, Bg, Br)),
    #                    cv2.merge((Lb, Lg, Lr)),
    #                    cv2.merge((Rb, Rg, Rr))]
    #     return self


if __name__ == '__main__':
    names = ['front','back', 'left', 'right']
    paramsfile = [os.path.join("my_yaml", name + ".yaml") for name in names]
    images = [os.path.join("und_smimages", name + ".png") for name in names]
    fisheyes = [Fisheye(p, n) for p,n in zip(paramsfile, names)]
    for i in range(4):
        fisheyes[i].build_undistort_map()
        fisheyes[i].build_align_map()
    undimages = [fisheyes[i].undistort(cv2.imread(image)) for i, image in enumerate(images)]
    aligns = [fisheyes[i].aling_project(undimage) for i, undimage in enumerate(undimages)]

    
    cys = []
    onecys = []
    for i in range(4):
        fisheyes[i].build_one_map()
        fisheyes[i].build_spherical_map()
        ROI_x,ROI_y = utils.fov_to_pixel(120,setting.sph_foc_len,setting.cx,setting.cy)
        cy = fisheyes[i].warp_spherical(aligns[i],ROI_y,ROI_x)
        onecy = fisheyes[i].warpone(cv2.imread(images[i]),ROI_x,ROI_y)
        cys.append(cy)
        onecys.append(onecy)
        #cv2.imshow(f"cy{i}", cv2.resize(cy, None, fx=0.9, fy=0.9))
        ROI_x,ROI_y = utils.fov_to_pixel(90,setting.sph_foc_len,setting.cx,setting.cy)
        cy = fisheyes[i].warp_spherical(aligns[i],ROI_y,ROI_x)
        #cv2.imshow(f"ccy{i}", cv2.resize(cy, None, fx=0.9, fy=0.9))
    panorama = Panorama(cys)
    cv2.imshow("F", cv2.resize(panorama.F, None, fx=0.9, fy=0.9))
    cv2.imshow("FL", cv2.resize(panorama.FL, None, fx=0.9, fy=0.9))
    cv2.imshow("FR", cv2.resize(panorama.FR, None, fx=0.9, fy=0.9))
    cv2.imshow("B", cv2.resize(panorama.B, None, fx=0.9, fy=0.9))
    cv2.imshow("BL", cv2.resize(panorama.BL, None, fx=0.9, fy=0.9))
    cv2.imshow("BR", cv2.resize(panorama.BR, None, fx=0.9, fy=0.9))
    cv2.imshow("L", cv2.resize(panorama.L, None, fx=0.9, fy=0.9))
    cv2.imshow("LB", cv2.resize(panorama.LF, None, fx=0.9, fy=0.9))
    cv2.imshow("LF", cv2.resize(panorama.LF, None, fx=0.9, fy=0.9))
    cv2.imshow("R", cv2.resize(panorama.R, None, fx=0.9, fy=0.9))
    cv2.imshow("RB", cv2.resize(panorama.RB, None, fx=0.9, fy=0.9))
    cv2.imshow("RF", cv2.resize(panorama.RF, None, fx=0.9, fy=0.9))
    cv2.waitKey(0)
