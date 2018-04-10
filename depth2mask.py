import warnings
warnings.simplefilter("ignore", UserWarning)

import numpy as np
olderr = np.seterr(all='ignore')
import cv2
from sys import argv
from os import listdir
from os.path import isfile, join
import imageio
from utils import checkDirAndCreate
from depth2HeightMskHelper import *
from labelHelper import *
import sys
sys.path.append('C:/Projects/semantic-segmentation')
from pspClassifier import *

from write2FileHelper import saveOpencvMatrix

class depth2maskTester(object):
    """docstring fordepth2maskTester."""
    def __init__(self, rootpath, srcImgPath, modelFilePath = None):
        self.depthHelper = depth2HeightMskHelper()
        self.classifier = pspClassifier(rootpath, srcImgPath, modelFile = modelFilePath)
        self.labelHelper = labelHelper(classifier = self.classifier)
    def fit(self,depthAddr = None, rawDepthAddr = None, camAddr=None, labelFile = None, imgName = None, forwardMethod = False):
        self.depthHelper.fit(depthAddr,rawDepthAddr,camAddr,forwardMethod=forwardMethod)
        if(forwardMethod and self.depthHelper.detectedBoxes == 0):
            return None
        self.labelHelper.fit(self.depthHelper, labelName = imgName, forwardMethod = forwardMethod)
    def save(self, obstacleName, heigtMapName):
        self.labelHelper.writeObstacles2File(obstacleName)
        saveOpencvMatrix(heigtMapName, self.depthHelper.heightMap)
    def plotAndOutput(self,imageName,heightMapFile = None,plotOnScreen=False,outImageWithBox=None, outForInputFile = None):
        # write heightMap without boundingbox, save as png image
        if(heightMapFile):
            imageio.imwrite(heightMapFile, self.depthHelper.heightMap)

        # plot final results on screen
        if(plotOnScreen):
            cv2.imshow("result", self.labelHelper.imageWithBox)
            cv2.waitKey(0)

        # write heightMap with boundingbox and rotatedBox, save as png image
        if(outImageWithBox):
            cv2.imwrite(outImageWithBox, self.labelHelper.imageWithBox)

        # write obstacles to file as input to cpp program as o:
        if(outForInputFile):
            self.labelHelper.writeObstacles2File(outForInputFile)

if __name__ == "__main__":
    rootpath = "C:/Projects/SUNRGB-dataset/_training/"
    modelFilePath = "C:/Projects/semantic-segmentation/pspnet_sunrgbd_sun_model2_resume.pkl"
    srcImgPath = rootpath+'imgs/'
    d2tTester = depth2maskTester(rootpath, srcImgPath, modelFilePath)
    # filenameSet = listdir(srcImgPath)
    #
    filenameSet = np.array([2276,2487,2501,2483])
    for name in filenameSet:
        # pureName = name.split('.')[0]
        depthAddr  = rootpath + 'depth/'+str(name)+'.png'
        rawDepthAddr = rootpath + 'depth_r/'+str(name)+'.png'
        d2tTester.fit(depthAddr = depthAddr, rawDepthAddr = rawDepthAddr, imgName = str(name) +'.jpg')
 # if __name__ == "__main__":
 #    rootpath = 'C:/Projects/SUNRGB-dataset/'
 #    labelFolderList = ['SUNRGBD-test_images/', 'testing/hha/']
 #    modelFilePath = "E:/pspnet_sunrgbd_sun_model.pkl"
 #
 #    d2tTester = depth2maskTester(rootpath,labelFolderList,modelFilePath)
 #
 #
 #    outputpath = 'imgs/'
 #    chooseSplit = "testing"
 #    resForInputFile = "layoutParam.txt"
 #    startIdx = 1861
 #    testList=np.array([2322])
 #    # testList=np.array([1970,1972,1975,2115,2243,2291,2293,2295,2297,2300,2321,2322,2330,2342,2348,2349,2352,2354,2377,2411,2441,2490])
 #    offsetTestList = testList - startIdx
 #    numOfTest = max(offsetTestList)
 #    olderr = np.seterr(all='ignore')
 #    try:
 #        fp = open(rootpath+'nyud2_testing.txt', 'r')
 #        filenameSet = fp.readlines()
 #    finally:
 #        fp.close()
 #    checkDirAndCreate(outputpath + chooseSplit, checkNameList = ['mask','res'])
 #
 #    for idx, file in enumerate(filenameSet):
 #        if(idx>numOfTest):
 #            break
 #        if(idx not in offsetTestList):
 #            continue
 #
 #        split_items = file.split('/')
 #        camAddr = rootpath + '/'.join(p for p in split_items[:-2]) + '/intrinsics.txt'
 #        depthAddr_root  = rootpath + '/'.join(p for p in split_items[:-2]) + '/depth_bfx/' #+ split_items[-1].split('.')[0]+'_abs.png'
 #        rawDepthAddr_root = rootpath + '/'.join(p for p in split_items[:-2]) + '/depth/' #+ split_items[-1].split('.')[0]+'_abs.png'
 #
 #        depthAddr = [depthAddr_root + f for f in listdir(depthAddr_root) if isfile(join(depthAddr_root,f ))][0]
 #        rawDepthAddr = [rawDepthAddr_root  +  f for f in listdir(rawDepthAddr_root) if isfile(join(rawDepthAddr_root,f ))][0]
 #
 #        heightFile = outputpath + chooseSplit+"/mask/"+str(idx+startIdx)+".png"
 #        resFile =  outputpath + chooseSplit+"/res/"+str(idx+startIdx)+".png"
 #        lFile = outputpath + '/pred/pred'+str(idx+startIdx) +'.png'
 #
 #        d2tTester.fit(depthAddr = depthAddr, rawDepthAddr = rawDepthAddr, camAddr=camAddr, imgName = str(idx+startIdx))
 #        d2tTester.plotAndOutput(str(idx+startIdx),heightMapFile = heightFile,plotOnScreen=False,outImageWithBox=resFile, outForInputFile = resForInputFile)
