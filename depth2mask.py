import numpy as np
import cv2
from sys import argv
import imageio
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from depthImgProcessor import processDepthImage
from camera import processCamMat

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

    roundX = X.astype(int)
    roundZ = Z.astype(int)
    maxX = np.max(roundX)
    maxZ = np.max(roundZ)
    minX = np.min(roundX)
    minZ = np.min(roundZ)

    x_range = maxX - minX + 1
    z_range = maxZ - minZ + 1
    
    mat_boundx = max(x_range, maxX+1)
    mat_boundz = max(z_range, maxZ+1)

    
    heightMap = np.ones([mat_boundz, mat_boundx]) * np.inf
    heightMap = heightMap.astype("float")
    for i in range(height):
        for j in range(width):
            tx = roundX[i,j]
            tz = mat_boundz - roundZ[i,j]
            heightMap[tz,tx] = min(h[i,j], heightMap[tz,tx])
    heightMap[np.where(heightMap==np.inf)] = 0
    heightMap = np.fliplr(heightMap)
    imgbounds = [minX, maxX, minZ, maxZ]
    return heightMap, imgbounds

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
    cv2.imshow('image', img)

def getObstacleMask(heightMap, area_threshold_min = 30, area_threshold_max_ratio = 0.9):
    minv = np.min(heightMap)
    vrange = np.max(heightMap) - minv
    heightMap = (heightMap-minv)/vrange * 255
    imgray = heightMap.astype("uint8")

    area_threshold_max = heightMap.shape[0] * heightMap.shape[1]*area_threshold_max_ratio
    im_denoise = cv2.fastNlMeansDenoising(imgray, None, 15, 7, 40)
    adp_thresh = cv2.adaptiveThreshold(im_denoise, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
    _, contours, _ = cv2.findContours(adp_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    clusterTooSmall = []
    boundingboxList = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if(area<area_threshold_min or area>area_threshold_max):
            clusterTooSmall.append(i)
        else:
            x,y,w,h = cv2.boundingRect(cnt)
            boundingboxList.append([x, y, x+w, y+h])
    # delete contours too small
    contours = np.delete(np.array(contours), clusterTooSmall)
    boundingboxes = np.array(boundingboxList)

    pickupIds, groupContents = non_max_supression(boundingboxes, 0.8)
    picked_boundingBox = boundingboxes[pickupIds]
    drawBoundingBox(imgray, picked_boundingBox)
    return picked_boundingBox

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

def main(root = 'C:/Projects/rgbd-processor-python/imgs/', depthAddr = "depth.png", rawDepthAddr = "rawdepth.png", camAddr = "intrinsics.txt", outfile = "autolay_input.txt"):
    with open(root+camAddr, 'r') as camf:
        cameraMatrix = processCamMat(camf.readlines())
    # cameraMatrix = np.array([[518.857901, 0.000000, 284.582449],[0.000000, 519.469611, 208.736166],[0.000000, 0.000000, 1.000000]])
    depthImage = imageio.imread(root+depthAddr).astype(float)/100
    rawDepth = imageio.imread(root+rawDepthAddr).astype(float)/1000
    missingMask = (rawDepth == 0)

    heightMap,imgbounds = getHeightMap(depthImage,rawDepth,cameraMatrix)
    obstacles = getObstacleMask(heightMap)
    writeObstacles2File(outfile, obstacles, imgbounds)
    cv2.waitKey(0)

if __name__ == "__main__":
    if(len(argv)<2):
        main()
    else:
        args = getopts(argv)
        main(root = args["--root"], depthAddr = args["--depth"], rawDepth = args["--raw"], camAddr=args["--cam"], outfile=args["--out"])
    # if '-i' in args:  # Example usage.
    #     print(myargs['-i'])
    