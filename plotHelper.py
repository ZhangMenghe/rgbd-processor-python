import numpy as np
import cv2
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def drawBoundingBox(imgray, rects):
    img = np.copy(imgray)
    if(len(imgray.shape)<3):
        img = cv2.cvtColor(imgray, cv2.COLOR_GRAY2BGR)
    for rect in rects:
        x1,y1,x2,y2 = rect
        cv2.rectangle(img, (x1,y1), (x2,y2), (0, 0, 255), 2)
    return img

def drawBoundingBoxAndRotatedBox(heightMap, boundingBoxes, rotatedBox):
    img_height = heightMap.astype("uint8")
    imageWithBox = drawBoundingBox(img_height, boundingBoxes)
    cv2.drawContours(imageWithBox, rotatedBox, -1, (255,255,0),2)
    cv2.imshow("result", imageWithBox)
    return imageWithBox
