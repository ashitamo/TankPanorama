import cv2
import numpy as np
from statistics import NormalDist

def gstreamer_pipeline(cam_id=0,
                       capture_width=960,
                       capture_height=640,
                       framerate=60,
                       flip_method=2):
    """
    Use libgstreamer to open csi-cameras.
    """
    return ("nvarguscamerasrc sensor-id={} ! ".format(cam_id) + \
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (capture_width,
               capture_height,
               framerate,
               flip_method
            )
    )


def convert_binary_to_bool(mask):
    """
    Convert a binary image (only one channel and pixels are 0 or 255) to
    a bool one (all pixels are 0 or 1).
    """
    return (mask.astype(np.float64) / 255.0).astype(int)


def adjust_luminance(gray, factor):
    """
    Adjust the luminance of a grayscale image by a factor.
    """
    return np.minimum((gray * factor), 255).astype(np.uint8)


def get_mean_statistisc(gray, mask):
    """
    Get the total values of a gray image in a region defined by a mask matrix.
    The mask matrix must have values either 0 or 1.
    """
    return np.sum(gray * mask)


def mean_luminance_ratio(grayA, grayB, mask):
    return get_mean_statistisc(grayA, mask) / get_mean_statistisc(grayB, mask)


def get_mask(img):
    """
    Convert an image to a mask array.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    return mask


def get_overlap_region_mask(imA, imB):
    """
    Given two images of the save size, get their overlapping region and
    convert this region to a mask array.
    """
    overlap = cv2.bitwise_and(imA, imB)
    mask = get_mask(overlap)
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=2)
    return mask


def get_outmost_polygon_boundary(img):
    """
    Given a mask image with the mask describes the overlapping region of
    two images, get the outmost contour of this region.
    """
    mask = get_mask(img)
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=2)
    cnts, hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)[-2:]
    # get the contour with largest aera
    C = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)[0]

    # polygon approximation
    polygon = cv2.approxPolyDP(C, 0.009 * cv2.arcLength(C, True), True)

    return polygon


def get_weight_mask_matrix(imA, imB, dist_threshold=5):
    """
    Get the weight matrix G that combines two images imA, imB smoothly.
    """

    width = imA.shape[1]
    kernel=np.ones((5, 5),np.uint8)
    # cv2.imshow('imA', cv2.resize(imA, None, fx=0.3, fy=0.3))
    # cv2.imshow('imB', cv2.resize(imB, None, fx=0.3, fy=0.3))
    overlapMask = get_overlap_region_mask(imA, imB)
    overlapMask = cv2.morphologyEx(overlapMask, cv2.MORPH_CLOSE, kernel, iterations=1)
    overlapMaskInv = cv2.bitwise_not(overlapMask)
    indices = np.where(overlapMask == 255)

    imA_diff = cv2.bitwise_and(imA, imA, mask=overlapMaskInv)
    imB_diff = cv2.bitwise_and(imB, imB, mask=overlapMaskInv)

    G = get_mask(imA).astype(np.float32) / 255.0
    G = cv2.morphologyEx(G, cv2.MORPH_CLOSE, kernel, iterations=1)
    polyA = get_outmost_polygon_boundary(imA_diff)
    polyB = get_outmost_polygon_boundary(imB_diff)
    for y, x in zip(*indices):

        #convert this x,y int an INT tuple
        xy_tuple = tuple([int(x), int(y)])
        #G[y,x] = (width-x)/width
        distToB = cv2.pointPolygonTest(polyB, xy_tuple, True)

        if distToB < dist_threshold:
            distToA = cv2.pointPolygonTest(polyA, xy_tuple, True)
            distToB *= distToB
            distToA *= distToA
            G[y, x] = distToB / (distToA + distToB)

    return G, overlapMask

def get_weight_mask_matrix_by_board_dist(imA, imB,reverse=False):
    """
    Get the weight matrix G that combines two images imA, imB smoothly.
    """

    width = imA.shape[1]
    kernel=np.ones((5, 5),np.uint8)

    overlapMask = get_overlap_region_mask(imA, imB)
    overlapMask = cv2.morphologyEx(overlapMask, cv2.MORPH_CLOSE, kernel, iterations=1)
    indices = np.where(overlapMask == 255)

    G = get_mask(imA).astype(np.float32) / 255.0
    G = cv2.morphologyEx(G, cv2.MORPH_CLOSE, kernel, iterations=1)

    u = 0.5
    sigma = 0.25
    nd = NormalDist(u, sigma)
    for y, x in zip(*indices):
        if reverse:
            t = (width-x)/width
        else:
            t = 1 - (width - x)/width
        G[y,x] = nd.cdf(t)


    return G, overlapMask

def make_white_balance(image):
    """
    Adjust white balance of an image base on the means of its channels.
    """
    B, G, R = cv2.split(image)
    m1 = np.mean(B)
    m2 = np.mean(G)
    m3 = np.mean(R)
    K = (m1 + m2 + m3) / 3
    c1 = K / m1
    c2 = K / m2
    c3 = K / m3
    B = adjust_luminance(B, c1)
    G = adjust_luminance(G, c2)
    R = adjust_luminance(R, c3)
    return cv2.merge((B, G, R))


def fov_to_pixel(fov,foc_len,cx,cy):
    '''
    parameters
        fov 目標FOV大小
        foc_len 焦距
        cx,cy 圖像中心
    return ROI_x ROI_y
        ROI_x[0] 為ROI左上角x座標
        ROI_y[0] 為ROI左上角y座標
        ROI_x[1] 為ROI右下角x座標
        ROI_y[1] 為ROI右下角y座標
    '''
    pix = fov/360*foc_len*np.pi
    ROI_x = [int(cx-pix),int(cx+pix)]
    ROI_y = [int(cy-pix),int(cy+pix)]
    return ROI_x,ROI_y

def dfov_to_pixel(fovl,fovr,foc_len,cx,cy,fovov):
    '''
    parameters
        fovl 目標FOV向左的大小
        fovr 目標FOV向右的大小
        foc_len 焦距
        cx,cy 圖像中心
    return ROI_x ROI_y
        ROI_x[0] 為ROI左上角x座標
        ROI_y[0] 為ROI左上角y座標
        ROI_x[1] 為ROI右下角x座標
        ROI_y[1] 為ROI右下角y座標
    '''
    pixl = fovl*2/360*foc_len*np.pi
    pixr = fovr*2/360*foc_len*np.pi
    pixov = int((fovov)/360*foc_len*np.pi)

    ROI_x = [int(cx-pixl),int(cx+pixr)]
    ROI_xovl = [ROI_x[0]-pixov,ROI_x[0]]
    ROI_xovr = [ROI_x[1],ROI_x[1]+pixov]
    return ROI_x,ROI_xovl,ROI_xovr


if __name__ == "__main__":
    G = np.ones((500, 200)) *255
    u = 0.5
    sigma = 0.25
    nd = NormalDist(u, sigma)
    print(nd.cdf(0))
    print(nd.cdf(0.5))
    print(nd.cdf(1))
    for y, x in zip(*np.where(G == 255)):
        t = (200 - x)/200
        # RC = 1
        # G[y,x] = -np.exp(RC*(t-1)) + 1
        #print(nd.cdf(-t))
        G[y,x] = nd.cdf(t)
    cv2.imshow("G", G)
    cv2.waitKey(0)