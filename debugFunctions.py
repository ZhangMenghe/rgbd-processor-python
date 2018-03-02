import cv2
import numpy as np
def debug_drawContoursOnDepthImg(depthImage, contours, obstaclBoxes, height2Img,img2Height):
    boundx = height2Img[-1]
    for idx, cnt in enumerate(contours):
        box = obstaclBoxes[idx]
        draw_list = []
        for tz in range(box[1], box[3]):
            for tx in range(box[0], box[2]):
                    draw_list.extend(height2Img[tz*boundx + tx])
        test = depthImage.flatten()
        test = np.copy(depthImage).astype(np.uint8).flatten()
        test[draw_list] = 255
        test=test.reshape(img2Height.shape)
        # imageio.imwrite(heightMapFile, heightMap)
        cv2.imshow("test"+str(idx), test)
        cv2.waitKey(0)
