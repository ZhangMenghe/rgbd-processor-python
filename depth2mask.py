import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np
import cv2
from sys import argv
from os import listdir
from os.path import isfile, join
import imageio
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from depthImgProcessor import processDepthImage
from camera import processCamMat
from utils import checkDirAndCreate

# Malisiewicz et al.
# reference:https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
def non_max_supression(rects, overlapThresh = 0.8):
    if(len(rects) == 0):
        return []
    boxes = rects.astype("float")

    pickIdx = []
    groupContents = []

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # find area and sort by y2
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while(len(idxs)>0):
        last = len(idxs) - 1
        i = idxs[last]
        pickIdx.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        group = np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        groupContents.append(group)
        idxs = np.delete(idxs, group)

    return pickIdx, groupContents

def plot3dHeightMap(mat_boundx, mat_boundz, heightMap):
    mx,mz = np.meshgrid(np.array(range(mat_boundx)), np.array(range(mat_boundz)))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(mx, mz, heightMap, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_xlabel('width')
    ax.set_ylabel('height')
    ax.set_zlabel('height above ground')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def getHeightMap(depthImage, missingMask, cameraMatrix):
    height, width = depthImage.shape
    pc, N, yDir, h, R = processDepthImage(depthImage, missingMask, cameraMatrix)

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
    img2Height = np.zeros(depthImage.shape, dtype=int)

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

def writeHeights2txtfile(outfilename, heightMap):
    with open(outfilename, "w")  as outfile:
        for row in heightMap:
            outfile.write(np.array_str(row).replace('\n', ''))
            outfile.write('\r\n')

def drawBoundingBox(imgray, rects):
    img = cv2.cvtColor(imgray, cv2.COLOR_GRAY2BGR)
    for rect in rects:
        x1,y1,x2,y2 = rect
        cv2.rectangle(img, (x1,y1), (x2,y2), (0, 0, 255), 2)
    return img
def getObstacleMask(heightMap, area_threshold_min_ratio = 0.005, area_threshold_max_ratio =0.9, needToDraw = False, needToStore = True):
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
    rotateboxList = []
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
            # rotated rectangle
            '''
            rect: a Box2D structure which contains :
            ( center (x,y), (width, height), angle of rotation ).
            But to draw this rectangle, we need 4 corners of the rectangle.
            It is obtained by the function cv2.boxPoints()
            '''
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            rotateboxList.append(box)
    # delete contours too small
    contours = np.delete(np.array(contours), clusterTooSmall)

    if(len(boundingboxList)==0):
        return contours, [], None
    boundingboxes = np.array(boundingboxList)
    rotateboxes = np.array(rotateboxList)
    pickupIds, groupContents = non_max_supression(boundingboxes, 0.8)
    contours = contours[pickupIds]
    picked_boundingBox = boundingboxes[pickupIds]
    # picked_boundingBox = boundingboxes
    img = drawBoundingBox(imgray, picked_boundingBox)
    img = cv2.drawContours(img, rotateboxes[pickupIds], -1, (0,255,0), 1)
    # img = drawRotatedBox(img, rotateboxList[pickupIds])
    if(needToDraw):
        cv2.imshow('image', img)
        cv2.waitKey(0)
    if(needToStore):
        return contours, picked_boundingBox, img
    return contours, picked_boundingBox, None

def getopts(argv):
    opts = {}  # Empty dictionary to store key-value pairs.
    while argv:  # While there are arguments left to parse...
        if argv[0][0] == '-':  # Found a "-name value" pair.
            opts[argv[0]] = argv[1]  # Add key and value to the dictionary.
        argv = argv[1:]  # Reduce the argument list by copying it starting from index 1.
    return opts

'''
bounds: minx, maxx, minz, maxz
'''
def writeObstacles2File(filename, boxes, bounds):
    boxes = boxes.astype("float")
    box_num = boxes.shape[0]
    widths = boxes[:,2] - boxes[:,0]
    heights = boxes[:,3] - boxes[:,1]
    cxs = (boxes[:,2] + boxes[:,0]) / 2 + bounds[0]
    cys = (boxes[:,3] + boxes[:,1]) / 2
    cys = -(bounds[3]/2 - (cys + bounds[2]))
    with open(filename, 'a') as fp:
        for i in range(box_num):
            fp.write('o : ' + str(cxs[i])+' ' + str(cys[i]) + ' 0 90 '+str(widths[i])+' '+str(heights[i]))
            fp.write('\r\n')

def setupInputMatrix(depthAddr, rawDepthAddr,camAddr):
    root = 'C:/Projects/rgbd-processor-python/imgs/'
    depthName = "depth.png"
    rawDepthName = "rawdepth.png"
    camName = "intrinsics.txt"
    depthAddr = root+depthName if depthAddr == None else depthAddr
    rawDepthAddr = root+rawDepthName if rawDepthAddr==None else rawDepthAddr
    camAddr = root+camName if camAddr == None else camAddr
    with open(camAddr, 'r') as camf:
        cameraMatrix = processCamMat(camf.readlines())
    # cameraMatrix = np.array([[518.857901, 0.000000, 284.582449],[0.000000, 519.469611, 208.736166],[0.000000, 0.000000, 1.000000]])
    depthImage = imageio.imread(depthAddr).astype(float)/100
    rawDepth = imageio.imread(rawDepthAddr).astype(float)/1000
    missingMask = (rawDepth == 0)
    return depthImage,missingMask,cameraMatrix
def getObstacleLabels(contours, obstaclBoxes, height2Img, img2Height, labelImgName = "label.png", fwRemovalRatio=0.8):
    labelImg = cv2.imread(labelImgName, 0)
    labelImg = cv2.resize(labelImg, (img2Height.shape[1], img2Height.shape[0]), interpolation=cv2.INTER_NEAREST).astype(int)
    heightMapMsk = []
    # heightMapMsk = np.zeros(height2Img.shape)
    clusterWrong = []
    boundx = height2Img[-1]
    for idx, cnt in enumerate(contours):
        box = obstaclBoxes[idx]
        check_img_loc_list = []
        for tz in range(box[1], box[3]):
            for tx in range(box[0], box[2]):
                    check_img_loc_list.extend(height2Img[tz*boundx + tx])

        checkImgColor = labelImg.flatten()[check_img_loc_list]
        # remove if wall and floor is accounts for the most part
        label_5 =len(np.where(checkImgColor ==5)[0])
        label_12 = len(np.where(checkImgColor == 12)[0])

        fwRatio = (label_5+label_12) / len(checkImgColor)
        print(fwRatio)
        if(fwRatio > fwRemovalRatio):
            clusterWrong.append(idx)
    print("----")
    contours = np.delete(contours, clusterWrong)
    obstacles =  np.delete(obstaclBoxes, clusterWrong)
    rotateboxList = []
    # consider how to join closing item?
    for i, cnt in enumerate(contours):
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        rotateboxList.append(box)
    # cv2.drawContours(heightMapMsk, rotateboxList, -1, 1)
    return obstacles, rotateboxList, heightMapMsk
def debug_drawContoursOnDepthImg(depthImage, contours, obstaclBoxes, height2Img,img2Height):
    boundx = height2Img[-1]
    for idx, cnt in enumerate(contours):
        box = obstaclBoxes[idx]
        draw_list = []
        for tz in range(box[1], box[3]):
            for tx in range(box[0], box[2]):
                    draw_list.extend(height2Img[tz*boundx + tx])

        # label_5 =len(np.where(checkImgColor == 5)[0])
        # label_12 = len(np.where(checkImgColor == 12)[0])

        test = depthImage.flatten()
        test = np.copy(depthImage).astype(np.uint8).flatten()
        test[draw_list] = 255
        test=test.reshape(img2Height.shape)
        # imageio.imwrite(heightMapFile, heightMap)
        cv2.imshow("test"+str(idx), test)
        cv2.waitKey(0)
def main(depthAddr = None, rawDepthAddr = None, camAddr=None, outfile = "autolay_input.txt", heightMapFile=None, resutlFile = None,labelFile = None):
    depthImage,missingMask,cameraMatrix = setupInputMatrix(depthAddr, rawDepthAddr,camAddr)
    heightMap,imgbounds, height2Img, img2Height = getHeightMap(depthImage,missingMask,cameraMatrix)
    if(heightMapFile != None):
        imageio.imwrite(heightMapFile, heightMap)
    contours, obstaclBoxes, imageWithBox= getObstacleMask(heightMap,needToDraw = False)
    if(len(obstaclBoxes) == 0):
        return
    # debug_drawContoursOnDepthImg(depthImage, contours, obstaclBoxes, height2Img,img2Height)
    # refined results with labels from NN result
    obstacles, rotatedBox, heightMapMsk = getObstacleLabels(contours, obstaclBoxes, height2Img, img2Height, labelImgName = labelFile)
    cv2.drawContours(imageWithBox, rotatedBox, -1, (255,255,0),2)
    cv2.imshow("result", imageWithBox)
    # if(resutlFile!=None):
    #     cv2.imwrite(resutlFile, imageWithBox)
    # writeObstacles2File(outfile, obstacles, imgbounds)
    cv2.waitKey(0)

if __name__ == "__main__":
    rootpath = 'C:/Projects/SUNRGB-dataset/'
    outputpath = 'imgs/'
    chooseSplit = "testing"
    startIdx =1861
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
        main(depthAddr, rawDepthAddr, camAddr, heightMapFile =heightFile,resutlFile=resFile, labelFile = lFile )
# if __name__ == "__main__":
#     if(len(argv)<2):
#         main()
#     else:
#         args = getopts(argv)
#         main(root = args["--root"], depthAddr = args["--depth"], rawDepth = args["--raw"], camAddr=args["--cam"], outfile=args["--out"])
#     # if '-i' in args:  # Example usage.
#     #     print(myargs['-i'])
