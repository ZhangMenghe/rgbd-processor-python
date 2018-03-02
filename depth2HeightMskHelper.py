import numpy as np
import cv2
from depthImgProcessor import processDepthImage
from utils import setupInputMatrix,non_max_supression

'''
Tackle with depth images ONLY and transform depth images to heigt mask
No labels needed for this phase
'''
class depth2HeightMskHelper(object):
    def __init__(self, depthAddr = None, rawDepthAddr = None, camAddr=None):
        self.depthImage, missingMask, cameraMatrix = setupInputMatrix(depthAddr, rawDepthAddr,camAddr)
        self.heightMap, self.imgbounds, self.height2Img, self.img2Height = self.getHeightMap(missingMask,cameraMatrix)

        self.contours, self.obstaclBoxes = self.getObstacleMask()
        self.detectedBoxes = len(self.obstaclBoxes)
    def getHeightMap(self, missingMask, cameraMatrix):
        height, width = self.depthImage.shape
        pc, N, yDir, h, R = processDepthImage(self.depthImage, missingMask, cameraMatrix)

        X = pc[:,:,0]
        Y = h
        Z = pc[:,:,2]

        # where each pixel will be located in 3d world
        roundX = X.astype(int)
        roundZ = Z.astype(int)
        maxX = np.max(roundX)
        maxZ = np.max(roundZ)
        minX = np.min(roundX)
        minZ = np.min(roundZ)
        # print(minX, maxX, minZ, maxZ)
        x_range = maxX - minX + 1
        z_range = maxZ - minZ + 1

        mat_boundx = max(x_range, maxX+1)
        mat_boundz = max(z_range, maxZ+1)


        heightMap = np.ones([mat_boundz, mat_boundx], dtype ="float") * np.inf

        # height2Img = np.zeros(heightMap.shape, dtype=int)
        height2Img = dict.fromkeys(range(len(heightMap.flatten())), [])
        height2Img[-1] = mat_boundx
        img2Height = np.zeros(self.depthImage.shape, dtype=int)

        for i in range(height):
            for j in range(width):
                tx = roundX[i,j] - minX
                tz = roundZ[i,j]
                # boudz-z cause we will flipup heightMap later
                idx_height = (mat_boundz - tz) * mat_boundx + tx
                img2Height[i,j] = idx_height
                if(height2Img[idx_height]):
                    height2Img[idx_height].append(i*width + j)
                else:
                    height2Img[idx_height] = [i*width + j]
                if h[i,j]<heightMap[tz,tx]:
                    heightMap[tz,tx] = h[i,j]
        heightMap[np.where(heightMap==np.inf)] = 0
        heightMap = np.flipud(heightMap)
        imgbounds = [minX, maxX, minZ, maxZ]
        return heightMap, imgbounds, height2Img, img2Height
    def getObstacleMask(self, area_threshold_min_ratio = 0.005, area_threshold_max_ratio =0.9):
        heightMap = np.copy(self.heightMap)
        minv = np.min(heightMap)
        vrange = np.max(heightMap) - minv
        heightMap = (heightMap-minv)/vrange * 255
        imgray = heightMap.astype("uint8")
        mapsize = heightMap.shape[0] * heightMap.shape[1]
        area_threshold_min = mapsize * area_threshold_min_ratio
        area_threshold_max = mapsize * area_threshold_max_ratio
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

        _, contours, _ = cv2.findContours(adp_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
            return [], []
        # Do non-maximum supression
        boundingboxes = np.array(boundingboxList)
        pickupIds, groupContents = non_max_supression(boundingboxes, 0.8)
        contours = contours[pickupIds]
        picked_boundingBox = boundingboxes[pickupIds]
        return contours, picked_boundingBox
