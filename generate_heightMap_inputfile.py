import numpy as np
import cv2
import imageio
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from depthImgProcessor import processDepthImage
from write2FileHelper import saveOpencvMatrix
def getHeightMap(depthImage, missingMask, cameraMatrix):
    height, width = depthImage.shape

    pc, N, yDir, h, R = processDepthImage(depthImage, missingMask, cameraMatrix)

    X = pc[:,:,0]
    Y = h
    Z = pc[:,:,2]
    #X = X - np.min(X) + 1
    # Z = Z - np.min(Z) + 1

    roundX = X.astype(int)
    roundZ = Z.astype(int)
    maxX = np.max(roundX)
    maxZ = np.max(roundZ)
    minX = np.min(roundX)
    minZ = np.min(roundZ)

    x_range = maxX - minX + 1
    z_range = maxZ - minZ + 1
    print(maxX)
    print(maxZ)
    print(minX)
    print(minZ)

    mat_boundx = max(x_range, maxX+1)
    mat_boundz = max(z_range, maxZ+1)

    mx,mz = np.meshgrid(np.array(range(mat_boundx)), np.array(range(mat_boundz)))
    heightMap = np.ones([mat_boundz, mat_boundx]) * np.inf
    for i in range(height):
        for j in range(width):
            tx = roundX[i,j]
            tz = roundZ[i,j]
            heightMap[tz,tx] = min(h[i,j], heightMap[tz,tx])
    heightMap[np.where(heightMap==np.inf)] = 0
    # heightMap = np.fliplr(heightMap)
    # colorMap = cv2.cvtColor(heightMap, cv2.COLOR_GRAY2BGR)
    # cv2.cvtcolor(heightMap, colorMap, COLOR_GRAY2BGR)
    imageio.imwrite('height2.png', heightMap)

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

    return heightMap

depthAddr = "imgs/depth.png"
rawDepthAddr = "imgs/rawdepth.png"
depthImage = imageio.imread(depthAddr).astype(float)
rawDepth = imageio.imread(rawDepthAddr).astype(float)/10
missingMask = (rawDepth == 0)
cameraMatrix = np.array([[518.857901, 0.000000, 284.582449],[0.000000, 519.469611, 208.736166],[0.000000, 0.000000, 1.000000]])
heightMap = getHeightMap(depthImage,rawDepth,cameraMatrix)
# saveOpencvMatrix("heightMapData.yml", heightMap)
# outfile = open("heightMaptxt.txt","w")
# for row in heightMap:
#     outfile.write(np.array_str(row).replace('\n', ''))
#     outfile.write('\r\n')
# outfile.close()
