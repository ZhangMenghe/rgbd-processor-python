import numpy as np
import cv2
import math
from depthImgProcessor import processDepthImage
from utils import setupInputMatrix,non_max_supression
from plotHelper import plot3dHeightMap
from utils import rotatePC
def getContourHeight(obstaclBoxes, heightMap):
    contourHeights = []
    for rect in obstaclBoxes:
        cx = int((rect[0] + rect[2])/2)
        cy = int((rect[1] + rect[3])/2)
        sampler = heightMap[cy-5:cy+5, cx-5:cx+5]
        contourHeights.append(np.average(sampler))
    return contourHeights
def getObstacleMask(mat):
    heightMap = np.copy(mat)
    minv = np.min(heightMap)
    vrange = np.max(heightMap) - minv
    heightMap = (heightMap-minv)/vrange * 255
    imgray = heightMap.astype(np.uint8)
    mapsize = heightMap.shape[0] * heightMap.shape[1]
    # print(mapsize)
    # cv2.imshow('ori', imgray)
    im_denoise = cv2.fastNlMeansDenoising(imgray, None, 15, 7, 40)
    ksize = int(mapsize/10000)
    if(ksize %2 == 0):
        ksize-=1
    im_median = cv2.medianBlur(im_denoise,ksize)
    # cv2.imshow('median', im_median)
    _,binary = cv2.threshold(im_median,50,255,cv2.THRESH_BINARY)
    # cv2.imshow('binary', binary)

    # structureElem = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, structureElem)
    # im_close = binary * mask
    kernel = np.ones((7,7),np.uint8)
    dilation = cv2.dilate(binary,kernel,iterations = 1)
    # cv2.imshow('dilation', dilation)


    adp_thresh = cv2.adaptiveThreshold(dilation, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)

    # cv2.imshow('thresh', adp_thresh)

    _, contours, _  = cv2.findContours(adp_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mapsize

def RemoveContourOverlapping(contours, mapsize, threshold = 0.8):
    area_threshold_min = mapsize * 0.005
    area_threshold_max = mapsize * 0.9
    clusterTooSmall = []
    boundingboxList = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if(area<area_threshold_min):
            clusterTooSmall.append(i)
        else:
            # print(area)
            x,y,w,h = cv2.boundingRect(cnt)
            if(w*h > area_threshold_max):
                clusterTooSmall.append(i)
                continue
            boundingboxList.append([x, y, x+w, y+h])

    # delete contours too small
    contours = np.delete(np.array(contours), clusterTooSmall)

    if(len(boundingboxList)==0):
        return [], [],[]
    # Do non-maximum supression
    boundingboxes = np.array(boundingboxList)
    pickupIds, groupContents = non_max_supression(boundingboxes, threshold)
    contours = contours[pickupIds]
    return contours, boundingboxes[pickupIds], pickupIds
'''
Tackle with depth images ONLY and transform depth images to heigt mask
No labels needed for this phase
'''
class depth2HeightMskHelper(object):
    def __init__(self):
        self.depthImage = None
        self.HHA = None
        self.heightMap = None
        #[minX, maxX, minZ, maxZ] in real world
        self.imgbounds = None
        self.heightMatBounds = None
        self.height2Img = None
        self.img2Height = None
        # contours in heightMap
        self.contours = None
        # height of each obstacle in heightMap
        self.contourHeights = None
        self.obstaclBoxes = None
        self.img2RealCoord = None
    def fit(self, depthAddr = None, rawDepthAddr = None, camAddr=None, generateHAA=True, forwardMethod = False):
        self.depthImage, missingMask, cameraMatrix = setupInputMatrix(depthAddr, rawDepthAddr, camAddr)
        self.getHeightMap(missingMask, cameraMatrix, generateHAA)
        if(forwardMethod):
            tmpContours, mapsize = getObstacleMask(self.heightMap)
            self.contours, self.obstaclBoxes,_ = RemoveContourOverlapping(tmpContours, mapsize)
            self.detectedBoxes = len(self.obstaclBoxes)
            self.contourHeights =  getContourHeight(self.obstaclBoxes, self.heightMap)
    def getHHAImage(self, pc, N, yDir, h):
        tmp = np.multiply(N, yDir)
        # with np.errstate(invalid='ignore'):
        acosValue = np.minimum(1,np.maximum(-1,np.sum(tmp, axis=2)))
        angle = np.array([math.degrees(math.acos(x)) for x in acosValue.flatten()])
        angle = np.reshape(angle, h.shape)

        pc[:,:,2] = np.maximum(pc[:,:,2], 100)
        I = np.zeros(pc.shape)
        I[:,:,0] = 31000/pc[:,:,2]
        I[:,:,1] = h
        I[:,:,2] = (angle + 128-90)
        self.HHA = I.astype(np.uint8)

    def getHeightMap(self, missingMask, cameraMatrix, generateHAA):
        height, width = self.depthImage.shape
        pc, N, yDir, h, R = processDepthImage(self.depthImage, missingMask, cameraMatrix)
        pcRot = rotatePC(pc, R)
        if(generateHAA):
            self.getHHAImage(pc, N, yDir, h)

        # where each pixel will be located in 3d world
        roundX = pcRot[:,:,0].astype(int)
        roundZ = pcRot[:,:,2].astype(int)

        # [minX, maxX, minZ, maxZ]
        self.imgbounds = [np.min(roundX), np.max(roundX), np.min(roundZ), np.max(roundZ)]

        x_range = self.imgbounds[1] - self.imgbounds[0] + 1
        z_range = self.imgbounds[3] - self.imgbounds[2] + 1

        mat_boundx = max(x_range, self.imgbounds[1]+1)
        mat_boundz = max(z_range, self.imgbounds[3]+1)


        self.heightMap = np.ones([mat_boundz, mat_boundx], dtype ="float") * np.inf

        self.height2Img = dict.fromkeys(range(len(self.heightMap.flatten())), [])

        self.img2Height = np.zeros((self.depthImage.shape[0], self.depthImage.shape[1], 2), dtype=int)

        for i in range(height):
            for j in range(width):
                tx = roundX[i,j] - self.imgbounds[0]
                tz = roundZ[i,j]
                self.img2Height[i,j] = [mat_boundz - tz, tx]
                # boudz-z cause we will flipup heightMap later
                idx_height = (mat_boundz - tz) * mat_boundx + tx
                if(self.height2Img[idx_height]):
                    self.height2Img[idx_height].append(i*width + j)
                else:
                    self.height2Img[idx_height] = [i*width + j]
                if h[i,j]<self.heightMap[tz,tx]:
                    self.heightMap[tz,tx] = h[i,j]
        self.heightMap[np.where(self.heightMap==np.inf)] = 0
        # plot3dHeightMap(mat_boundx,mat_boundz,self.heightMap)
        self.heightMap = np.flipud(self.heightMap)
        self.heightMatBounds = [mat_boundz, mat_boundx]
