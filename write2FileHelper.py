import numpy as np
import cv2
def writeHeights2txtfile(outfilename, heightMap):
    with open(outfilename, "w")  as outfile:
        for row in heightMap:
            outfile.write(np.array_str(row).replace('\n', ''))
            outfile.write('\r\n')
def saveOpencvMatrix(filename, matrix):
    ofile = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
    ofile.write("floatdata", matrix)
    ofile.release()
