import numpy as np
import cv2
from scipy import stats
import matplotlib.pyplot as plt
from utils import rotateAroundPoint
from plotHelper import drawBoundingBox
from depth2HeightMskHelper import getObstacleMask, RemoveContourOverlapping, getContourHeight

'''
This class is used to optimize those obstacle from heightMap by classification results
'''
class labelHelper(object):
    """docstring for labelHelper."""
    def __init__(self, classifier = None, fwRemovalRatio = 0.8, shrinkF = 1.5):
        # imgBounds: minx, maxx, minz, maxz
        self.imgbounds = None
        self.contours = None
        self.height2Img = None
        self.img2Height = None
        self.classifier = classifier
        self.fwRemovalRatio = fwRemovalRatio
        # refined results with labels from NN result
        self.boundingBoxes = None
        self.boxesFromDepth = None
        self.rotatedBox = None
        self.rotatedRect = None
        self.heightMapMsk = None
        self.boxLabel = None
        self.mergedLables = None
        self.mergeIdx = None
        self.imageWithBox = None
        self.sceneMatList = None
        self.sceneMat = None
        self.shrinkF = shrinkF
        self.cate = ["unknown", "floor ","sofa ","chair ","bed ","NightStand","shelf","table","wall","onwallObjs","otherFurniture","ceiling"]
        self.labelColor = {0:[0,0,0], 1:[173,216,230], 2:[139, 0 ,139], 3:[255,0,0], 4:[156, 156, 156], 5:[0,255,0],\
        6:[255,165,0], 7:[173,255,47],8:[255, 228, 225],9:[159, 121, 238],10:[139,69,0],11:[255,106,106],12:[0,0,255],13:[255,2552,255]}

        # self.labelDict = {"bed":1, "books":2, "ceiling":3, "chair":4, "floor":5, "furniture":6, "objects":7, "pics":8, "sofa":9, "table":10, "tv":11, "wall":12, "window":13 }
        # self.getObstacleLabels()

    def fit(self, depthHelper,labelName=None, labelFile=None, forwardMethod = True):
        self.boxLabel = []
        self.mergeIdx = []
        self.rotatedBox = []
        self.rotatedRect = []
        self.mergedLables = []
        self.sceneMatList =[]
        self.imgbounds = depthHelper.imgbounds
        self.contours = depthHelper.contours
        self.contourHeights = depthHelper.contourHeights
        self.height2Img = depthHelper.height2Img
        self.img2Height = depthHelper.img2Height
        self.boxesFromDepth = depthHelper.obstaclBoxes
        self.HHA = depthHelper.HHA
        self.heightMap = depthHelper.heightMap
        self.img2RealCoord = depthHelper.img2RealCoord
        self.heightMapMsk = np.zeros(depthHelper.heightMatBounds, dtype=np.uint8)
        self.sceneMat = np.zeros([int(self.heightMap.shape[0]/self.shrinkF)+1,int(self.heightMap.shape[1]/self.shrinkF)+1 ])
        self.camCenter = (int(-self.imgbounds[0]- self.sceneMat.shape[1]/2), int(self.sceneMat.shape[0] / 2 - (self.imgbounds[3]- self.imgbounds[2])))
        # self.getObstacleLabels(labelName, labelFile)
        self.combineHeightAndLabel(labelName, labelFile, forwardMethod)
    def combineHeightAndLabel(self, labelName, labelFile, forwardMethod):
        if(self.classifier and labelName):
            labelImg = self.classifier.fit(labelName,self.HHA)
        elif(labelFile):
            labelImg = cv2.imread(labelFile, 0)
        else:
            return False
        labelImg = cv2.resize(labelImg, (self.img2Height.shape[1], self.img2Height.shape[0]), interpolation=cv2.INTER_NEAREST).astype(int)
        decoded = self.classifier.dataset.decode_segmap(labelImg)
        # cv2.imshow("resized", decoded)
        # cv2.waitKey(0)
        if(forwardMethod):
            self.getObstacleLabels(labelImg)
        else:
            self.getObstacleRealWorld(labelImg)
    def fitWall(self, xdata, ydata):
        samplerLoc = np.where(np.logical_and(xdata<25, xdata>-25))
        sx, sy = xdata[samplerLoc], ydata[samplerLoc]
        slope, intercept, r_value, p_value, std_err = stats.linregress(sx, sy)
        angle = np.arctan(slope)
        cs,se = np.cos(angle),np.sin(angle)
        self.alignWallAngel = (-se, cs)
        # x_rot,y_rot = rotateAroundPoint(self.camCenter, np.vstack([xdata,ydata]), -se, cs)
        # x_rots,y_rots = rotateAroundPoint(self.camCenter, np.vstack([sx,sy]), -se, cs)
        # slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(x_rots, y_rots)
        # plt.plot(x_rot, y_rot, 'o', label='original data')
        # plt.plot(sx, intercept + slope*sx, 'r', label='fitted line')
        # plt.legend()
        # plt.show()
    def getObstacleRealWorld(self, labelImg):
        denoteList =[n for n in np.unique(labelImg) if n not in [1,8,9,11]]
        # WALL
        wallMat = np.zeros(self.sceneMat.shape)
        ybound, xbound = self.sceneMat.shape[0] /2, self.sceneMat.shape[1] /2
        xdata,ydata = [],[]
        for cate in [8,9]:
            mappedLoc = self.img2Height[np.where(labelImg==cate)]#self.img2RealCoord[floorLocs]
            for loc in mappedLoc:
                xdata.append(int(loc[1]/self.shrinkF))
                ydata.append(int(loc[0]/self.shrinkF))
                wallMat[ydata[-1],xdata[-1]] = 255
        xdata = (np.array(xdata) - xbound).astype(int)
        ydata = (ybound - np.array(ydata)).astype(int)

        cv2.imshow("wall", wallMat)
        # cv2.waitKey(0)
        self.fitWall(xdata, ydata)

        for cate in denoteList:
            mappedLoc = self.img2Height[np.where(labelImg==cate)]#self.img2RealCoord[floorLocs]
            for loc in mappedLoc:
                self.sceneMat[int(loc[0]/self.shrinkF),int(loc[1]/self.shrinkF)] = cate

        # logicalRes = np.logical_and(self.sceneMat, self.heightMap)
        # self.sceneMat[logicalRes == 0] = 0
        self.sceneMat = cv2.resize(self.sceneMat, (self.heightMap.shape[1],self.heightMap.shape[0]), interpolation=cv2.INTER_NEAREST)

        for cate in denoteList:
            cate_sceneMat = np.zeros(self.sceneMat.shape)
            cate_sceneMat[self.sceneMat == cate] = 1
            # cv2.imshow("cate-"+str(cate),cate_sceneMat)
            # cv2.waitKey(0)
            self.sceneMatList.append(cate_sceneMat)
        self.boxLabel = denoteList

        # cv2.imshow("label",self.sceneMat)
        # cv2.imshow("height",self.heightMap)
        # cv2.waitKey(0)
        self.getObstacleMaskFromLabeledMask()

    def getObstacleMaskFromLabeledMask(self):
        labels = []
        contours = []
        boundingBoxes = []
        pickupIds = []
        mapsize = self.sceneMat.shape[0] * self.sceneMat.shape[1]
        for i, cate_sceneMat in enumerate(self.sceneMatList):
            # if(self.boxLabel[i] == 4):
            imgray = cate_sceneMat.astype(np.uint8)
            im_close = cv2.morphologyEx(imgray, cv2.MORPH_CLOSE,  cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))
            _, cate_contours, _  = cv2.findContours(im_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cate_contours,cate_boundings,cate_ids = RemoveContourOverlapping(cate_contours,mapsize,threshold=0.5)
            contours.extend(cate_contours)
            boundingBoxes.extend(cate_boundings)
            pickupIds.extend(cate_ids)
            labels.extend([ self.boxLabel[i] ]* len(cate_contours))
        if(len(contours) > 2):
            contours,boundingBoxes,pickupIds = RemoveContourOverlapping(contours,mapsize,threshold=0.5)
            self.boxLabel = np.array(labels)[pickupIds]
        else:
            self.boxLabel = np.array(labels)

        for i, cnt in enumerate(contours):
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            self.rotatedBox.append(box)
            self.rotatedRect.append(rect)
        self.contourHeights =  getContourHeight(boundingBoxes, self.heightMap)

        print("labels-" + str(self.boxLabel))
        rgbView =  np.zeros([self.sceneMat.shape[0],self.sceneMat.shape[1],3],dtype= np.uint8)
        for cate in np.unique(self.sceneMat):
            rgbView[self.sceneMat == cate] = self.labelColor[cate]
        res = cv2.drawContours(rgbView, self.rotatedBox, -1,  (0,255,0), 2)

        # cv2.circle(rgbView, (int(self.sceneMat.shape[1]/2), int(self.imgbounds[3] - self.imgbounds[2])), 5, (0,0,255))
        # cv2.imshow("res", rgbView)
        # cv2.waitKey(0)
    def getObstacleLabels(self, labelImg):
        keepCluster = []
        boundx = self.heightMapMsk.shape[1]
        for idx, cnt in enumerate(self.contours):
            box = self.boxesFromDepth[idx]
            check_img_loc_list = []
            for tz in range(box[1], box[3]):
                for tx in range(box[0], box[2]):
                    check_img_loc_list.extend(self.height2Img[tz*boundx+ tx])

            checkImgColor = labelImg.flatten()[check_img_loc_list]

            # remove if wall and floor is accounts for the most part
            label_5 = (checkImgColor == 5).sum()
            label_12 = (checkImgColor == 12).sum()
            fwRatio = (label_5+label_12) / len(checkImgColor)
            if(fwRatio > self.fwRemovalRatio):
                continue

            # decide the obj type in this bounding box
            objlabel = -1
            deciderCount = 0
            for label in np.unique(checkImgColor):
                if(label == 5 or label == 12 or label == 0):
                    continue
                if((checkImgColor==label).sum() > deciderCount):
                    objlabel = label
                    deciderCount = (checkImgColor==label).sum()
            self.boxLabel.append(objlabel)
            keepCluster.append(idx)
        #now use label, to decide whether to merge those boundingboxes
        self.mergeObjects(self.boxesFromDepth[keepCluster])
        self.contourHeights = np.array(self.contourHeights)[keepCluster]
        self.contours = self.contours[keepCluster]
        # rotated rectangle
        '''
        rect: a Box2D structure which contains :
        ( center (x,y), (width, height), angle of rotation ).
        But to draw this rectangle, we need 4 corners of the rectangle.
        It is obtained by the function cv2.boxPoints()
        '''
        for i, cnt in enumerate(self.contours):
            rect = cv2.minAreaRect(cnt)
            # box point: bl, tl, tr, br
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            self.rotatedBox.append(box)
            self.rotatedRect.append(rect)
        cv2.drawContours(self.heightMapMsk, self.rotatedBox, -1, 255, thickness=-1)
        # get image with boxes
        self.getImageWithBox()
        return True
    def getImageWithBox(self):
        self.imageWithBox = np.copy(self.heightMap)
        self.imageWithBox = self.imageWithBox.astype(np.uint8)
        cv2.drawContours(self.imageWithBox, self.rotatedBox, -1, 255, thickness=1)
        self.imageWithBox = drawBoundingBox(self.imageWithBox, self.boundingBoxes)
        cv2.drawContours(self.imageWithBox, self.rotatedBox, -1, (255,255,0), 2)
    def getBoxInfo(self, boxes):
        centerx = (boxes[:,2] + boxes[:,0])/2.0
        centery = (boxes[:,3] + boxes[:,1])/2.0
        r =np.sqrt ((boxes[:,2] - boxes[:,0])**2 + (boxes[:,3] - boxes[:,1])**2)/2.0
        return np.vstack([centerx, centery, r]).T
    def merge2Boxes(self, box, cbox):
        return [min(box[0], cbox[0]), min(box[1], cbox[1]),max(box[2],cbox[2]), max(box[3],cbox[3])]
    def deleteTooClose(self, boxesInfo,ratioThresh = 1):
        for i, box in enumerate(boxesInfo):
            for j in range(i+1, len(boxesInfo)):
                cbox = boxesInfo[j]
                ratio = (box[2]+cbox[2])/(np.sqrt((box[0]-cbox[0])**2+(box[1]-cbox[1])**2))
                # print(ratio)
                if(ratio>ratioThresh):
                    return [i,j]
        return None
    def checkAndMergeBoxes(self, boundingboxes):
        boxes = np.copy(boundingboxes)
        recordLst = [[x] for x in range(len(boxes))]
        boxesInfo = self.getBoxInfo(boxes)
        while(True):
            deleteLst = self.deleteTooClose(boxesInfo)
            if(deleteLst == None):
                return boxes, recordLst

            if(len(recordLst)==2):
                recordLst = [recordLst[0] + recordLst[1]]
            else:
                mergedItem = recordLst.pop(deleteLst[0])
                mergedItem += recordLst.pop(deleteLst[1])
                recordLst += [mergedItem]
            mergeBox = self.merge2Boxes(boxes[deleteLst[0]],boxes[deleteLst[1]])
            boxes = np.delete(boxes,deleteLst,axis=0)
            boxes = np.vstack([boxes, mergeBox])
            boxesInfo = np.delete(boxesInfo, deleteLst, axis=0)
            boxesInfo = np.vstack([boxesInfo, self.getBoxInfo(np.array([mergeBox]))])
    def mergeObjects(self, boundingboxes, thresh = 30):
        #check those boxed very closed to each other
        numOfBox = len(boundingboxes)

        numOfLabel = len(np.unique(self.boxLabel))

        mergedBoxes = []
        mergedLables = []
        if(numOfLabel == numOfBox):
            self.boundingBoxes = boundingboxes
            return
        for label in np.unique(self.boxLabel):
            index = np.where(self.boxLabel==label)
            if(len(index[0]) == 1):
                mergedBoxes.extend(boundingboxes[index])
                mergedLables.append(label)
                continue
            addToMergedBoxes, mergedList = self.checkAndMergeBoxes(boundingboxes[index])
            mergedBoxes.extend(addToMergedBoxes)
            mergedLables.extend(len(addToMergedBoxes) * [label])
            for lst in mergedList:
                if(len(lst)!=1):
                    self.mergeIdx.append(index[0][lst])
        self.boundingBoxes = np.array(mergedBoxes)
        self.mergedLables = mergedLables
    def alignWall(self):
        rgbView =  np.zeros([self.sceneMat.shape[0],self.sceneMat.shape[1],3],dtype= np.uint8)
        rotBoxes = []
        rotRects = []
        alignAngle = np.rad2deg(np.arctan(self.alignWallAngel[0]/ self.alignWallAngel[1]))
        ybound, xbound = self.sceneMat.shape[0] /2, self.sceneMat.shape[1] /2
        for i,box in enumerate(self.rotatedBox):
            xdata = (np.array(box).T[0] - xbound).astype(int)
            ydata = (ybound - np.array(box).T[1]).astype(int)
            xrot, yrot = rotateAroundPoint(self.camCenter, np.vstack([xdata,ydata]), self.alignWallAngel[0], self.alignWallAngel[1])
            xrot = xrot+xbound
            yrot = ybound - yrot
            rotBoxes.append(np.vstack([xrot,yrot]).astype(int).T)
            rotRects.append([(xrot[0] + xrot[2])/2, (yrot[0] + yrot[2])/2, self.rotatedRect[i][1][0], self.rotatedRect[i][1][1], -(self.rotatedRect[i][2] + alignAngle]))
        self.alignedRotBox = rotBoxes
        self.alignedRotRect = rotRects
        res = cv2.drawContours(rgbView, rotBoxes, -1,  (0,255,0), 2)
        cv2.imshow("aftRot", rgbView)
        cv2.waitKey(0)
    def writeObstacles2File(self, filename):
        self.alignWall()
        rotatedBox = np.array(self.rotatedBox, dtype=float)
        prefix = 'objFixed : '
        prefix2 = 'group: '
        vertexIdx = [0,3,2,1]
        with open(filename, 'w') as fp:
            for i, rect in enumerate(self.rotatedRect):
                content  = prefix
                # fill vertices
                vertices = self.rotatedBox[i]
                for idx in vertexIdx:
                    content+=str(vertices[idx][0]) + ' ' + str(vertices[idx][1])+ ' '
                # center, size, angle
                for idx, item in enumerate(rect):
                    if(idx<2):
                        content += str(item[0]) + ' ' + str(item[1]) + ' '
                    else:
                        content += str(item) + ' '

                content += str(self.boxLabel[i]) + ' ' + str(self.contourHeights[i])
                fp.write(content +"\r\n")
            for lst in self.mergeIdx:
                content = prefix2
                for item in lst:
                    content += str(item) + ' '
                fp.write(content + '\r\n')
