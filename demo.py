
import scipy.io as spio
import numpy as np
import imageio
import math
import os
from os import listdir
from os.path import isfile, join
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from camera import cropCamera,getCameraParam
from depthImgProcessor import processDepthImage


def getHHAImg(depthImage, missingMask,cameraMatrix):
    pc, N, yDir, h, R = processDepthImage(depthImage * 100, missingMask, cameraMatrix)

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
    HHA = I.astype(np.uint8)
    return HHA

def getHeightMap(depthImage, missingMask,cameraMatrix):
    height, width = depthImage.shape

    pc, N, yDir, h, R = processDepthImage(depthImage, missingMask, cameraMatrix)

    X = pc[:,:,0]
    Y = h
    Z = pc[:,:,2]
    X = X - np.min(X) + 1
    Z = Z - np.min(Z) + 1
    roundX = X.astype(int)
    roundZ = Z.astype(int)
    maxX = np.max(roundX)
    maxZ = np.max(roundZ)
    mx,mz = np.meshgrid(np.array(range(maxX+1)), np.array(range(maxZ+1)))
    heightMap = np.ones([maxZ+1, maxX+1]) * np.inf
    for i in range(height):
        for j in range(width):
            tx = roundX[i,j]
            tz = roundZ[i,j]
            heightMap[tz,tx] = min(h[i,j], heightMap[tz,tx])
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

def processCamMat(strList):
    if(len(strList) == 1):
        numbers = np.array([strList[0].split(' ')[:9]]).astype(float)
        mat = np.reshape(numbers,[3,3], 'C')
    else:
        mat = np.zeros([3,3])
        for idx, line in enumerate(strList):
            line = line.rstrip() #rstrip() returns a copy of the string in which all chars have been stripped from the end of the string (default whitespace characters).
            numbers = line.split(' ')
            mat[idx,:] = np.array([numbers]).astype(float)
    return mat
def checkDirAndCreate(folder):
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    if not os.path.exists(folder + '/hha/'):
        try:
            os.makedirs(folder + '/hha/')
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    if not os.path.exists(folder + '/height/'):
      try:
          os.makedirs(folder + '/height/')
      except OSError as exc: # Guard against race condition
          if exc.errno != errno.EEXIST:
              raise
def main():
    rootpath = 'C:/Projects/SUNRGB-dataset/'
    outputpath = 'imgs/'
    chooseSplit = "testing"
    olderr = np.seterr(all='ignore')
    try:
        fp = open(rootpath+'sunrgbd_'+chooseSplit+'_images.txt', 'r')
        filenameSet = fp.readlines()
    finally:
        fp.close()
    checkDirAndCreate(outputpath + chooseSplit)
    for idx, file in enumerate(filenameSet):
        split_items = file.split('/')
        camAddr = rootpath + '/'.join(p for p in split_items[:-2]) + '/intrinsics.txt'
        with open(camAddr, 'r') as camf:
            cameraMatrix = processCamMat(camf.readlines())

        depthAddr_root  = rootpath + '/'.join(p for p in split_items[:-2]) + '/depth_bfx/' #+ split_items[-1].split('.')[0]+'_abs.png'
        rawDepthAddr_root = rootpath + '/'.join(p for p in split_items[:-2]) + '/depth/' #+ split_items[-1].split('.')[0]+'_abs.png'

        depthAddr = [depthAddr_root + f for f in listdir(depthAddr_root) if isfile(join(depthAddr_root,f ))][0]
        rawDepthAddr = [rawDepthAddr_root  +  f for f in listdir(rawDepthAddr_root) if isfile(join(rawDepthAddr_root,f ))][0]

        depthImage = imageio.imread(depthAddr).astype(float)/10000
        rawDepth = imageio.imread(rawDepthAddr).astype(float)/100000
        missingMask = (rawDepth == 0)

        HHA = getHHAImg(depthImage, missingMask, cameraMatrix)

        imageio.imwrite(outputpath + chooseSplit + '/hha/' + str(idx+1) + '.png',HHA)
        imageio.imwrite(outputpath + chooseSplit + '/height/' + str(idx+1) + '.png', HHA[:,:,1])

if __name__ == "__main__":
    main()

# def main():
#     rootpath = 'C:\Projects\SUNRGB-dataset\'
#     chooseSplit = "training"
#     olderr = np.seterr(all='ignore')
#     try:
#         fp = open(rootpath+'sunrgbd_'+chooseSplit+'_images.txt', 'r')
#         filenameSet = fp.readlines()
#     finally:
#         fp.close()
#     dataset_set = ['kinect2data', 'align_kv2', 'NYUdata', 'b3dodata', 'sun3ddata','xtion_align_data','realsense']
#
#
#
#
#     depthImage = imageio.imread("imgs/depth.png").astype(float)/1000
#     rawDepth = imageio.imread("imgs/rawdepth.png").astype(float)
#     missingMask = (rawDepth == 0)
#
#     # getHeightMap(depthImage, missingMask)
#     HHA = getHHAImg(depthImage, missingMask, cameraMatrix)
#     imageio.imwrite(outputpath + 'hha/' + str(idx) + '.png',HHA)
#     imageio.imwrite(outputpath + 'height' + .png', HHA[:,:,1])
# if __name__ == "__main__":
#     main()
