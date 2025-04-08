import cv2
import numpy as np
import setting
from surround_view import utils
import time
import threading
import queue
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import sys
class Fisheye(threading.Thread):
    def __init__(self,paramsfile,camera_name,cap_number = None):
        self.paramsfile = paramsfile
        self.camera_name = camera_name
        self.K = None
        self.D = None
    
        self.NEW_K = None
        self.NEW_W = None
        self.NEW_H = None
        self.scale_xy = None
        self.shift_xy = None
        self.project_matrix = None
        self.shp_K = None
        self.ROI_x,_ = utils.fov_to_pixel(setting.sfovx,setting.sph_foc_len,setting.cx,setting.cy)
        _,self.ROI_y = utils.fov_to_pixel(setting.sfovy,setting.sph_foc_len,setting.cx,setting.cy)
        #self.ROI_y = [0,setting.targeth]
        self.load_params(paramsfile)
        
        self.map_x = None
        self.map_y = None
        self.align_map_x = None
        self.align_map_y = None
        self.sphmap_x = None
        self.sphmap_y = None

        super(Fisheye, self).__init__()
        self.stopflag = False
        self.queue_in = queue.Queue(1)
        self.queue_out = queue.Queue(1)
        
        self.cap_number = cap_number
        if self.cap_number is not None:
            if sys.platform == "win32":
                self.cap = cv2.VideoCapture(self.cap_number, cv2.CAP_DSHOW)
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) # auto mode
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # manual mode
                self.cap.set(cv2.CAP_PROP_EXPOSURE, -11)
                # self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2); 
            else:
                self.cap = cv2.VideoCapture(self.cap_number,cv2.CAP_V4L2)
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) # auto mode
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # manual mode
                self.cap.set(cv2.CAP_PROP_EXPOSURE, 200)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2); 
        
            desired_width = 640 # 你想設定的寬度
            desired_height = 480  # 你想設定的高度
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
            desired_fps = 30  # 你想設定的FPS值
            self.cap.set(cv2.CAP_PROP_FPS, desired_fps)

            

    def load_params(self, paramsfile):
        fs = cv2.FileStorage(paramsfile, cv2.FILE_STORAGE_READ)
        self.K = fs.getNode("camera_matrix").mat()
        self.D = fs.getNode("dist_coeffs").mat()
        self.NEW_K = fs.getNode("new_camera_matrix").mat()
        self.scale_xy = fs.getNode("scale_xy").mat()
        self.shift_xy = fs.getNode("shift_xy").mat()
        self.project_matrix = fs.getNode("project_matrix").mat()
        self.align_project_matrix = fs.getNode("align_project_matrix").mat()
        self.shp_K = fs.getNode("spherical_camera_matrix").mat()
        fs.release()

    def initnewK(self):
        self.NEW_K = self.K.copy()
        self.set_shift(self.shift_xy)
        self.NEW_K[0,2] = setting.targetw//2
        self.NEW_K[1,2] = setting.targeth//2
        self.NEW_W = setting.targetw
        self.NEW_H = setting.targeth

    def build_undistort_map(self):
        indy, indx = np.indices((setting.targeth, setting.targetw), dtype=np.float32)
        objp = np.array([(indx.ravel()-self.NEW_K[0,2])/self.NEW_K[0,0], (indy.ravel()-self.NEW_K[1,2])/self.NEW_K[1,1], np.zeros_like(indx).ravel()]).T.reshape(setting.targeth, setting.targetw,3)
        rvec = np.array([[[0., 0., 0.]]])
        tvec = np.array([[[0., 0., 0.]]])
        imgpoints2, _ = cv2.fisheye.projectPoints(objp, rvec, tvec, self.K, self.D)
        self.map_x,self.map_y = imgpoints2[:, :, 0], imgpoints2[:, :, 1]
        #self.map_x,self.map_y = cv2.convertMaps(self.map_x, self.map_y, cv2.CV_32FC1,cv2.CV_32FC1)
        #self.map_x,self.map_y = cv2.fisheye.initUndistortRectifyMap(self.K,self.D,np.eye(3),self.NEW_K,(self.NEW_W, self.NEW_H),cv2.CV_16SC2)

    def warpone(self, image):
        return cv2.remap(image, self.onemap_x[self.ROI_y[0]:self.ROI_y[1],self.ROI_x[0]:self.ROI_x[1]], self.onemap_y[self.ROI_y[0]:self.ROI_y[1],self.ROI_x[0]:self.ROI_x[1]], cv2.INTER_LINEAR)

    def build_one_map(self):
        foc_len = (self.shp_K[0][0] + self.shp_K[1][1])/2
        temp = np.mgrid[0:setting.targetw,0:setting.targeth]
        x,y = temp[0],temp[1]
        phi = (x - self.shp_K[0][2])/foc_len # angle phi
        theta = (y - self.shp_K[1][2])/foc_len # theta
        p = np.array([np.cos(theta) * np.sin(phi) , np.sin(theta),np.cos(theta)*np.cos(phi)]).T.reshape(-1,3)
        image_points = self.shp_K.dot(p.T).T
        points = image_points[:,:]/image_points[:,[-1]]

        
        H = np.linalg.inv(self.align_project_matrix)
        lin_homg_ind = np.array([points[:,0].ravel(), points[:,1].ravel(), np.ones_like(points[:,0]).ravel()])
        map_ind = H.dot(lin_homg_ind)
        map_x, map_y = map_ind[:-1]/map_ind[-1]  # ensure homogeneity
        map_x, map_y = map_x.reshape(setting.targeth, setting.targetw).astype(np.float32), map_y.reshape(setting.targeth, setting.targetw).astype(np.float32)
        
        objp = np.array([(map_x.ravel()-self.NEW_K[0,2])/self.NEW_K[0,0], (map_y.ravel()-self.NEW_K[1,2])/self.NEW_K[1,1], np.zeros_like(map_x).ravel()]).T.reshape(setting.targeth, setting.targetw,3)
        rvec = np.array([[[0., 0., 0.]]])
        tvec = np.array([[[0., 0., 0.]]])
        imgpoints2, _ = cv2.fisheye.projectPoints(objp, rvec, tvec, self.K, self.D)
        self.onemap_x,self.onemap_y = imgpoints2[:, :, 0].astype(np.float32), imgpoints2[:, :, 1].astype(np.float32)
        self.onemap_x,self.onemap_y = cv2.convertMaps(self.onemap_x, self.onemap_y, cv2.CV_32FC1,cv2.CV_32FC1)

    def undistort(self, image):
        return cv2.remap(image, self.map_x, self.map_y, cv2.INTER_LINEAR)

    def set_shift(self, shift):
        self.shift_xy = shift
        self.NEW_K[0,2] = setting.targetw//2 + shift[0]
        self.NEW_K[1,2] = setting.targeth//2 + shift[1]
    
    def build_align_map(self):
        H = np.linalg.inv(self.align_project_matrix)
        indy, indx = np.indices((setting.targeth, setting.targetw), dtype=np.float32)
        lin_homg_ind = np.array([indx.ravel(), indy.ravel(), np.ones_like(indx).ravel()])
        map_ind = H.dot(lin_homg_ind)
        map_x, map_y = map_ind[:-1]/map_ind[-1]  # ensure homogeneity
        self.align_map_x = map_x.reshape(setting.targeth, setting.targetw).astype(np.float32)
        self.align_map_y = map_y.reshape(setting.targeth, setting.targetw).astype(np.float32)
        self.align_map_x, self.align_map_y = cv2.convertMaps(self.align_map_x, self.align_map_y, cv2.CV_32FC1, cv2.CV_32FC1)

    def aling_project(self, image):
        #self.set_shift(self.shift_xy)
        return cv2.remap(image, self.align_map_x, self.align_map_y, cv2.INTER_LINEAR)
    
    def tobirdview(self,image):
        return cv2.warpPerspective(image, self.project_matrix, (1800*2,850*2))
    
    
    def build_spherical_map(self):
        foc_len = (self.shp_K[0][0] + self.shp_K[1][1])/2
        temp = np.mgrid[0:setting.targetw,0:setting.targeth]
        x,y = temp[0],temp[1]
        phi = (x - self.shp_K[0][2])/foc_len # angle phi
        theta = (y - self.shp_K[1][2])/foc_len # theta
        p = np.array([np.cos(theta) * np.sin(phi) , np.sin(theta),np.cos(theta)*np.cos(phi)]).T.reshape(-1,3)
        image_points = self.shp_K.dot(p.T).T
        points = image_points[:,:-1]/image_points[:,[-1]]
        points = points.reshape(setting.targeth,setting.targetw,-1)
        self.sphmap_x = (points[:, :, 0]).astype(np.float32)
        self.sphmap_y = (points[:, :, 1]).astype(np.float32)
        self.sphmap_x, self.sphmap_y = cv2.convertMaps(self.sphmap_x, self.sphmap_y, cv2.CV_32FC1, cv2.CV_32FC1)

    def warp_spherical(self,img):
        return cv2.remap(img, self.sphmap_x[self.ROI_y[0]:self.ROI_y[1],self.ROI_x[0]:self.ROI_x[1]], self.sphmap_y[self.ROI_y[0]:self.ROI_y[1],self.ROI_x[0]:self.ROI_x[1]], cv2.INTER_LINEAR)

    def drawForwadLine(self,image,camera=None,t=5):
        def getLine(start,end,s=600,st = 100):
            l = end - start
            l = l / np.linalg.norm(l)
            ss = np.arange(0, s, st)
            ss = np.vstack((ss, ss)).T
            l = ss*l
            l[:,0] = l[:,0] + start[0]
            l[:,1] = l[:,1] + start[1]
            #cv2.polylines(image, [l.astype(np.int32)], False, (255, 0, 0), 5)
            return l[:,0],l[:,1]
        def draw(image,x,y):
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

        if camera is None:
            camera = self
        x,y = getLine(np.array([600, int(setting.targeth)+250]),
                        np.array([setting.cx, setting.cy])
                    )
        rx,ry = getLine(np.array([int(setting.targetw)-600, int(setting.targeth)+250]),
                        np.array([setting.cx, setting.cy])
                    )
        xl1,yl1 = getLine(np.array([x[2],y[2]]),
                        np.array([x[2] + 100 ,y[2]]),
                        s = 430,
                        st = 10
                    )
        xl2,yl2 = getLine(np.array([x[4],y[4]]),
                        np.array([x[4] + 50 ,y[4]]),
                        s = 250,
                        st = 10
                    )
        draw(image,x,y)
        draw(image,rx,ry)
        draw(image,xl1,yl1)
        draw(image,xl2,yl2)
    
    def save_data(self):
        fs = cv2.FileStorage(self.paramsfile, cv2.FILE_STORAGE_WRITE)
        fs.write("camera_matrix", self.K)
        fs.write("new_camera_matrix", self.NEW_K)
        fs.write("dist_coeffs", self.D)
        fs.write("project_matrix", self.project_matrix)
        fs.write("scale_xy", np.float32(self.scale_xy))
        fs.write("shift_xy", np.float32(self.shift_xy))
        fs.write("align_project_matrix", np.float32(self.align_project_matrix))
        if self.shp_K is not None:
            fs.write("spherical_camera_matrix", np.float32(self.shp_K))
        fs.release()
    def run(self):
        while not self.stopflag:
            if self.cap_number is not None:
                _,image = self.cap.read()
            else:
                try:
                    image = self.queue_in.get(timeout=0.1)
                except queue.Empty:
                    continue
            if image is not None:
                if self.cap_number is not None:
                    image = self.warpone(image)
                else:
                    image = self.warpone(cv2.imread(image))
                # if self.camera_name == "front":
                #     self.drawForwadLine(image)
                if self.queue_out.full():
                    self.queue_out.get()
                    # print("get")
                self.queue_out.put(image)


            
if __name__ == "__main__":
    name = "back"
    yaml = "my_yaml/" + name + ".yaml"
    image = "und_smimages/" + name + ".png"
    setting.sfovx = 90
    fisheye = Fisheye(yaml, name)
    fisheye.build_spherical_map()
    #fisheye.set_shift((0,0))
    fisheye.build_undistort_map()
    fisheye.build_align_map()
    image = cv2.imread(image)
    last = time.time()
    undistorted = fisheye.undistort(image)
    align = fisheye.aling_project(undistorted)
    cy = fisheye.warp_spherical(align)
    print("多次投影:", (time.time()-last))

    cv2.imshow("undistorted", cv2.resize(undistorted, None, fx=0.8, fy=0.8))
    cv2.imshow("align", cv2.resize(align, None, fx=0.8, fy=0.8))
    cv2.imshow("cy", cv2.resize(cy, None, fx=0.9, fy=0.9))

    fisheye.build_one_map()
    last = time.time()
    cc = fisheye.warpone(image)
    print("一次投影:", (time.time()-last))
    cv2.imshow("cc", cv2.resize(cc, None, fx=0.9, fy=0.9))

    cv2.waitKey(0)