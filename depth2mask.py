import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np
import cv2
from sys import argv
from os import listdir
from os.path import isfile, join
import imageio
from utils import checkDirAndCreate
from write2FileHelper import writeObstacles2File
from plotHelper import drawBoundingBoxAndRotatedBox
from depth2HeightMskHelper import *
from labelHelper import *










# def main(depthAddr = None, rawDepthAddr = None, camAddr=None, outfile = "autolay_input.txt", heightMapFile=None, resutlFile = None,labelFile = None):
#     depthImage,missingMask,cameraMatrix = setupInputMatrix(depthAddr, rawDepthAddr,camAddr)
#     heightMap,imgbounds, height2Img, img2Height = getHeightMap(depthImage,missingMask,cameraMatrix)
#     img_height = heightMap.astype("uint8")
#     if(heightMapFile != None):
#         imageio.imwrite(heightMapFile, heightMap)
#     contours, obstaclBoxes = getObstacleMask(heightMap)
#     if(len(obstaclBoxes) == 0):
#         return
#     # refined results with labels from NN result
#     obstacles, rotatedBox, heightMapMsk = getObstacleLabels(contours, obstaclBoxes, height2Img, img2Height, labelImgName = labelFile)
#     # print("obstacles" + str(obstacles.shape))
#     imageWithBox = drawBoundingBox(img_height, obstacles)
#     cv2.drawContours(imageWithBox, rotatedBox, -1, (255,255,0),2)
#     cv2.imshow("result", imageWithBox)
#     if(resutlFile != None):
#         cv2.imwrite(resutlFile, imageWithBox)
#     writeObstacles2File(outfile, obstacles, imgbounds)
#     cv2.waitKey(0)

if __name__ == "__main__":
    rootpath = 'C:/Projects/SUNRGB-dataset/'
    outputpath = 'imgs/'
    chooseSplit = "testing"
    startIdx = 1861
    #testList=np.array([1970,1972])
    testList=np.array([1970,1972,1975,2115,2243,2291,2293,2295,2297,2300,2321,2322,2330,2342,2348,2349,2352,2354,2377,2411,2441,2490])
    offsetTestList = testList - startIdx
    numOfTest = max(offsetTestList)
    olderr = np.seterr(all='ignore')
    try:
        fp = open(rootpath+'nyud2_testing.txt', 'r')
        filenameSet = fp.readlines()
    finally:
        fp.close()
    checkDirAndCreate(outputpath + chooseSplit, checkNameList = ['mask','res'])
    for idx, file in enumerate(filenameSet):
        if(idx>numOfTest):
            break
        if(idx not in offsetTestList):
            continue
        split_items = file.split('/')
        camAddr = rootpath + '/'.join(p for p in split_items[:-2]) + '/intrinsics.txt'
        depthAddr_root  = rootpath + '/'.join(p for p in split_items[:-2]) + '/depth_bfx/' #+ split_items[-1].split('.')[0]+'_abs.png'
        rawDepthAddr_root = rootpath + '/'.join(p for p in split_items[:-2]) + '/depth/' #+ split_items[-1].split('.')[0]+'_abs.png'

        depthAddr = [depthAddr_root + f for f in listdir(depthAddr_root) if isfile(join(depthAddr_root,f ))][0]
        rawDepthAddr = [rawDepthAddr_root  +  f for f in listdir(rawDepthAddr_root) if isfile(join(rawDepthAddr_root,f ))][0]
        heightFile = outputpath + chooseSplit+"/mask/"+str(idx+startIdx)+".png"
        resFile =  outputpath + chooseSplit+"/res/"+str(idx+startIdx)+".png"
        lFile = outputpath + '/pred/pred'+str(idx+startIdx) +'.png'
        # main(depthAddr, rawDepthAddr, camAddr, heightMapFile =heightFile,resutlFile=resFile, labelFile = lFile )
        mdepthHelper = depth2HeightMskHelper(depthAddr, rawDepthAddr, camAddr)
        mlabelHelper = labelHelper(mdepthHelper, labelFile = lFile)
        drawBoundingBoxAndRotatedBox(mlabelHelper.heightMapMsk, mlabelHelper.boundingBoxes, mlabelHelper.rotatedBox)

# if __name__ == "__main__":
#     if(len(argv)<2):
#         main()
#     else:
#         args = getopts(argv)
#         main(root = args["--root"], depthAddr = args["--depth"], rawDepth = args["--raw"], camAddr=args["--cam"], outfile=args["--out"])
#     # if '-i' in args:  # Example usage.
#     #     print(myargs['-i'])
