import numpy as np
import cv2
from plotHelper import drawBoundingBox
'''
This class is used to optimize those obstacle from heightMap by classification results
'''
class labelHelper(object):
    """docstring for labelHelper."""
    def __init__(self, classifier = None, fwRemovalRatio = 0.8):
        # imgBounds: minx, maxx, minz, maxz
        self.imgBounds = None
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
        # self.getObstacleLabels()

    def fit(self, depthHelper,labelName=None, labelFile=None, forwardMethod = True):
        self.boxLabel = []
        self.mergeIdx = []
        self.rotatedBox = []
        self.rotatedRect = []
        self.mergedLables = []
        self.imgBounds = depthHelper.imgbounds
        self.contours = depthHelper.contours
        self.contourHeights = depthHelper.contourHeights
        self.height2Img = depthHelper.height2Img
        self.img2Height = depthHelper.img2Height
        self.boxesFromDepth = depthHelper.obstaclBoxes
        self.HHA = depthHelper.HHA
        self.heightMap = depthHelper.heightMap
        self.heightMapMsk = np.zeros(depthHelper.heightMatBounds, dtype=np.uint8)
        # self.getObstacleLabels(labelName, labelFile)
        self.combineHeightAndLabel(labelName, labelFile, forwardMethod)
    def combineHeightAndLabel(self, labelName, labelFile, forwardMethod):
        if(self.classifier and labelName):
            labelImg = self.classifier.fit(labelName)
        elif(labelFile):
            labelImg = cv2.imread(labelFile, 0)
        else:
            return False
        labelImg = cv2.resize(labelImg, (self.img2Height.shape[1], self.img2Height.shape[0]), interpolation=cv2.INTER_NEAREST).astype(int)
        if(forwardMethod):
            self.getObstacleLabels(labelImg)
        else:
            self.getObstacleRealWorld(labelImg)
    def getObstacleRealWorld(self, labelImg):
        #todo
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

    def writeObstacles2File(self, filename):
        boxes = self.boundingBoxes.astype("float")
        rotatedBox = np.array(self.rotatedBox, dtype=float)
        prefix = 'objFixed : '
        prefix2 = 'group: '
        vertexIdx = [0,3,2,1]
        with open(filename, 'a') as fp:
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
