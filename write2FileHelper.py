import numpy as np
def writeHeights2txtfile(outfilename, heightMap):
    with open(outfilename, "w")  as outfile:
        for row in heightMap:
            outfile.write(np.array_str(row).replace('\n', ''))
            outfile.write('\r\n')
