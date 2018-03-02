import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np
import cv2
from sys import argv
from os import listdir
from os.path import isfile, join
import imageio
from utils import checkDirAndCreate
from plotHelper import drawBoundingBoxAndRotatedBox
from depth2HeightMskHelper import *
from labelHelper import *

def main(depthAddr = None, rawDepthAddr = None, camAddr=None, outfile = "autolay_input.txt", heightMapFile=None, resutlFile = None,labelFile = None):
    mdepthHelper = depth2HeightMskHelper(depthAddr, rawDepthAddr, camAddr)
    # write heightMap without boundingbox, save as png image
    if(heightMapFile):
        imageio.imwrite(heightMapFile, mdepthHelper.heightMap)
    if(mdepthHelper.detectedBoxes == 0):
        return
    mlabelHelper = labelHelper(mdepthHelper, labelFile = lFile)

    # plot final results on screen
    imageWithBox = drawBoundingBoxAndRotatedBox(mlabelHelper.heightMapMsk, mlabelHelper.boundingBoxes, mlabelHelper.rotatedBox)

    # write heightMap with boundingbox and rotatedBox, save as png image
    if(resutlFile):
        cv2.imwrite(resutlFile, imageWithBox)

    # write obstacles to file as input to cpp program as o:
    if(outfile):
        mlabelHelper.writeObstacles2File(outfile)
    cv2.waitKey(0)

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

        main(depthAddr, rawDepthAddr, camAddr, heightMapFile = heightFile, resutlFile=resFile, labelFile = lFile )

# if __name__ == "__main__":
#     if(len(argv)<2):
#         main()
#     else:
#         args = getopts(argv)
#         main(root = args["--root"], depthAddr = args["--depth"], rawDepth = args["--raw"], camAddr=args["--cam"], outfile=args["--out"])
#     # if '-i' in args:  # Example usage.
#     #     print(myargs['-i'])
