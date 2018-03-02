import numpy as np
import cv2
'''
This class is used to optimize those obstacle from heightMap by classification results
'''
class labelHelper(object):
    """docstring for labelHelper."""
    def __init__(self, depthHelper, classifier = None, labelFile= None, fwRemovalRatio = 0.8):
        if(len(depthHelper.obstaclBoxes) == 0):
            return
        # imgBounds: minx, maxx, minz, maxz
        self.imgBounds = depthHelper.imgbounds
        self.contours = depthHelper.contours
        self.height2Img = depthHelper.height2Img
        self.img2Height = depthHelper.img2Height
        self.classifier = classifier
        self.labelFile = labelFile
        self.fwRemovalRatio = fwRemovalRatio
        # refined results with labels from NN result
        self.boundingBoxes, self.rotatedBox, self.heightMapMsk = self.getObstacleLabels(depthHelper.obstaclBoxes)
        self.heightMapMsk = depthHelper.heightMap

    def getObstacleLabels(self, obstaclBoxes):
        labelImg = cv2.imread(self.labelFile, 0)
        labelImg = cv2.resize(labelImg, (self.img2Height.shape[1], self.img2Height.shape[0]), interpolation=cv2.INTER_NEAREST).astype(int)
        heightMapMsk =[]
        keepCluster = []
        boundx = self.height2Img[-1]
        boxLabel = []
        for idx, cnt in enumerate(self.contours):
            box = obstaclBoxes[idx]
            check_img_loc_list = []
            for tz in range(box[1], box[3]):
                for tx in range(box[0], box[2]):
                    check_img_loc_list.extend(self.height2Img[tz*boundx + tx])

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
            boxLabel.append(objlabel)
            keepCluster.append(idx)
        #now use label, to decide whether to merge those boundingboxes

        obstacles = self.mergeObjects(obstaclBoxes[keepCluster], boxLabel)
        # print("after Merge" + str(len(obstacles)))
        # print(obstacles.shape)
        # obstacles  =   obstaclBoxes[keepCluster]
        # print(obstacles.shape)
        self.contours = self.contours[keepCluster]
        rotateboxList = []
        # rotated rectangle
        '''
        rect: a Box2D structure which contains :
        ( center (x,y), (width, height), angle of rotation ).
        But to draw this rectangle, we need 4 corners of the rectangle.
        It is obtained by the function cv2.boxPoints()
        '''
        for i, cnt in enumerate(self.contours):
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            rotateboxList.append(box)
        # cv2.drawContours(heightMapMsk, rotateboxList, -1, 1)
        return obstacles, rotateboxList, heightMapMsk
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
    def checkAndMergeBoxes(self, boxes):
        boxesInfo = self.getBoxInfo(boxes)
        while(True):
            deleteLst = self.deleteTooClose(boxesInfo)
            if(deleteLst == None):
                return boxes
            mergeBox = self.merge2Boxes(boxes[deleteLst[0]],boxes[deleteLst[1]])
            boxes = np.delete(boxes,deleteLst,axis=0)
            boxes = np.vstack([boxes, mergeBox])
            boxesInfo = np.delete(boxesInfo, deleteLst, axis=0)
            boxesInfo = np.vstack([boxesInfo, self.getBoxInfo(np.array([mergeBox]))])
    def mergeObjects(self, boundingboxes, boxLabel, thresh = 30):
        #check those boxed very closed to each other
        numOfBox = len(boundingboxes)
        numOfLabel = len(np.unique(boxLabel))
        mergedBoxes = []
        if(numOfLabel == numOfBox):
            return boundingboxes
        # print("boxLabels : " + str(boxLabel))
        for label in np.unique(boxLabel):
            index = np.where(boxLabel==label)
            if(len(index[0]) == 1):
                mergedBoxes.extend(boundingboxes[index])
                continue
            mergedBoxes.extend(self.checkAndMergeBoxes(boundingboxes[index]))
        return np.array(mergedBoxes)
    def writeObstacles2File(self, filename):
        boxes = self.boundingBoxes.astype("float")
        box_num = boxes.shape[0]
        widths = boxes[:,2] - boxes[:,0]
        heights = boxes[:,3] - boxes[:,1]
        cxs = (boxes[:,2] + boxes[:,0]) / 2 + self.imgBounds[0]
        cys = (boxes[:,3] + boxes[:,1]) / 2
        cys = -(self.imgBounds[3]/2 - (cys + self.imgBounds[2]))
        with open(filename, 'a') as fp:
            for i in range(box_num):
                fp.write('o : ' + str(cxs[i])+' ' + str(cys[i]) + ' 0 90 '+str(widths[i])+' '+str(heights[i]))
                fp.write('\r\n')
