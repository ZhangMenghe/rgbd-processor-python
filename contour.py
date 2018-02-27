import numpy as np
import cv2

def non_max_supression(rects, overlapThresh):
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
		# print(overlap)
		# delete all indexes from the index list that have
		group = np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
		groupContents.append(group)
		idxs = np.delete(idxs, group)	

	return pickIdx, groupContents

area_threshold_min =30

im = cv2.imread('height.png')
area_threshold_max = im.shape[0] * im.shape[1]*0.9
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# Try opening and closing??
# ret, thresh = cv2.threshold(imgray, 127, 255,3)

# se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
# se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
# mask = cv2.morphologyEx(im, cv2.MORPH_CLOSE, se1)
# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
# im_refine = im * mask
# thresh = cv2.Canny(im_refine, 100, 200)


im_refine = cv2.fastNlMeansDenoising(imgray, None,15,7,40)
thresh = cv2.adaptiveThreshold(im_refine,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)
# cv2.imshow('MORPH RESULT', thresh)

im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
clusterTooSmall = []
hullList = []
boundingboxList = []
rotateboxList = []
for i, cnt in enumerate(contours):
	# moments
	M = cv2.moments(cnt)
	area = cv2.contourArea(cnt)
	if(area<area_threshold_min or area>area_threshold_max):
		clusterTooSmall.append(i)
	else:
		# hull = cv2.convexHull(cnt)
		# hullList.append(hull)
		x,y,w,h = cv2.boundingRect(cnt)
		boundingboxList.append([x, y, x+w, y+h])
		# rotated rectangle
		# rect = cv2.minAreaRect(cnt)
		# box = cv2.boxPoints(rect)
		# box = np.int0(box)
		# rotateboxList.append(box)
contours = np.delete(np.array(contours), clusterTooSmall)
# print(len(contours))
boundingboxes = np.array(boundingboxList)
pickupIds,groupContents = non_max_supression(boundingboxes, 0.8)
# print(pickupIds)
# print(groupContents)
# im3 = cv2.drawContours(im, rotateboxList, -1, (0,255,0), 1)
for rect in boundingboxes[pickupIds]:
	x1,y1,x2,y2 = rect
	print(x1,y1,x2,y2)
	cv2.rectangle(im, (x1,y1), (x2,y2), (0, 0, 255), 2)

# cnt = contours[4]
# im3 = cv2.drawContours(im, [cnt], 0, (0,255,0), 1)
cv2.imshow('image', im)
cv2.waitKey(0)