import numpy as np 
def writeHeights2txtfile(outfilename, heightMap):
    with open(outfilename, "w")  as outfile:
        for row in heightMap:
            outfile.write(np.array_str(row).replace('\n', ''))
            outfile.write('\r\n')
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
